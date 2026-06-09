#!/usr/bin/env python3
"""
同篇论文多表数据联动检查
================================

检测同一篇论文中，不同表格的同一组命名（如"Control组""空白组"）数值是否一致。
这是论文内部矛盾——同一篇论文中相同实验组的数值不应该在不同表格中不同。

三种检测模式：
1. 精确匹配：完全相同列名 → 跨表比对数值
2. 前缀匹配："Control" vs "Control组" vs "空白对照组" → 模糊匹配
3. Mean±SD一致性：同一列在两表中 Mean 差 > 2×SD 池 → 矛盾

原理：
- 多表数据常来自同一个实验，Control 组的 Mean ± SD 应一致
- 如果不一致，说明数据有问题——最常见的是复制数据忘了改 Control 列
- 与其他检测器不同，这是论文内部矛盾，不需要阴性对照

用法：
    python3 cross_table_consistency.py 论文.pdf [--output report.json]

依赖: PyMuPDF (fitz), numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class TableGroup:
    """一张表中的一组数据"""
    table_id: str          # "表1" "表2" 等
    group_name: str        # "Control" "阴性对照组" 等
    mean_val: float
    sd_val: float
    n_val: Optional[int] = None
    raw_text: str = ""     # 原始文本片段


@dataclass
class CrossTableFinding:
    """跨表不一致发现"""
    group_label: str       # 组的标准化名称
    table_a: str
    table_b: str
    mean_a: float
    mean_b: float
    sd_a: float
    sd_b: float
    z_score: float         # 差异的 Z 分数
    p_value: float         # 差异的显著性
    severity: str          # HIGH/MEDIUM/LOW
    description: str


# ═══════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════

class CrossTableConsistencyChecker:
    """
    同篇论文多表数据联动检查器

    检测同一篇论文中不同表格的同一组命名数值是否一致。
    论文内部矛盾 = 最强证据级别。
    """

    # 组名规范化映射
    GROUP_ALIASES = {
        # Control 系列
        "control": "Control",
        "control组": "Control",
        "空白对照": "Control",
        "空白对照组": "Control",
        "阴性对照": "Control",
        "阴性对照组": "Control",
        "nc": "Control",
        "ctrl": "Control",
        # Treatment 系列
        "treatment": "Treatment",
        "treatment组": "Treatment",
        "给药组": "Treatment",
        "药物组": "Treatment",
        # 模型系列
        "model": "Model",
        "模型组": "Model",
        # siRNA 系列
        "sirna": "siRNA",
        "si-": "siRNA",
        "sh-": "shRNA",
        # 常用缩写
        "dll4": "DLL4",
        "vegf": "VEGF",
        "pbs": "PBS",
    }

    def __init__(self):
        self.data_groups: Dict[str, List[TableGroup]] = defaultdict(list)
        self.findings: List[CrossTableFinding] = []

    def extract_from_pdf(self, pdf_path: str) -> Dict[str, List[TableGroup]]:
        """从PDF提取所有表格组数据"""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz) 未安装")

        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        # 找所有含有 Mean±SD 或 mean±SD 数据行
        # 模式: 组名 + 数字 ± 数字
        # 例如: Control  1.23 ± 0.45
        pattern = re.compile(
            r'(?P<group>[\u4e00-\u9fff\w\s\-/]+?)'  # 组名
            r'\s+'
            r'(?P<mean>\d+\.?\d*)\s*±\s*(?P<sd>\d+\.?\d*)'  # mean ± sd
        )

        # 按表分段
        table_pattern = re.compile(r'(?:表\s*\d+|Table\s*\d+|Tab\.?\s*\d+)')

        current_table = "未知表"
        table_positions = [(m.group(), m.start()) for m in table_pattern.finditer(full_text)]

        for match in pattern.finditer(full_text):
            group_raw = match.group('group').strip()
            mean_val = float(match.group('mean'))
            sd_val = float(match.group('sd'))

            # 判断当前属于哪个表
            pos = match.start()
            for t_name, t_pos in reversed(table_positions):
                if t_pos < pos:
                    current_table = t_name
                    break

            # 规范化组名
            group_normalized = self._normalize_group_name(group_raw)

            self.data_groups[group_normalized].append(TableGroup(
                table_id=current_table,
                group_name=group_raw,
                mean_val=mean_val,
                sd_val=sd_val,
                raw_text=match.group()
            ))

        doc.close()
        return dict(self.data_groups)

    def _normalize_group_name(self, name: str) -> str:
        """规范化组名"""
        name_lower = name.lower().strip()
        for alias, canonical in self.GROUP_ALIASES.items():
            if alias in name_lower:
                return canonical
        return name

    def check_consistency(self) -> List[CrossTableFinding]:
        """检查所有同组跨表数据一致性"""
        findings = []

        for group_name, entries in self.data_groups.items():
            if len(entries) < 2:
                continue  # 只在一个表中出现，无法比对

            # 两两比较同一组在不同表中的数值
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    a, b = entries[i], entries[j]
                    if a.table_id == b.table_id:
                        continue  # 同一张表内不比较（可能是不同时间点）

                    mean_diff = abs(a.mean_val - b.mean_val)
                    pooled_sd = np.sqrt((a.sd_val**2 + b.sd_val**2) / 2)

                    if pooled_sd < 1e-10:
                        pooled_sd = 1e-10  # 防止除零

                    z_score = mean_diff / pooled_sd

                    # 使用 Welch-Satterthwaite 近似做 t 检验
                    # 假设每组 n=3（保守估计）
                    n_eff = 3
                    se = np.sqrt(a.sd_val**2 / n_eff + b.sd_val**2 / n_eff)
                    if se > 0:
                        t_stat = mean_diff / se
                        df = n_eff * 2 - 2
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                    else:
                        t_stat = 0
                        p_val = 1.0

                    # 判定严重程度
                    if z_score > 3.0 and p_val < 0.01:
                        severity = "HIGH"
                    elif z_score > 2.0 and p_val < 0.05:
                        severity = "MEDIUM"
                    else:
                        severity = "LOW"

                    if severity == "LOW":
                        continue  # 差异不显著，不报告

                    findings.append(CrossTableFinding(
                        group_label=group_name,
                        table_a=a.table_id,
                        table_b=b.table_id,
                        mean_a=a.mean_val,
                        mean_b=b.mean_val,
                        sd_a=a.sd_val,
                        sd_b=b.sd_val,
                        z_score=round(z_score, 2),
                        p_value=round(p_val, 6),
                        severity=severity,
                        description=(
                            f"同组'{group_name}'在{a.table_id}和{b.table_id}中数据不一致："
                            f"Mean={a.mean_val}±{a.sd_val} vs {b.mean_val}±{b.sd_val}, "
                            f"z={z_score:.2f}, p={p_val:.4f}"
                        )
                    ))

        self.findings = findings
        return findings

    def to_dict(self) -> Dict:
        """导出结果"""
        return {
            "total_groups": len(self.data_groups),
            "groups_with_conflicts": len(set(f.group_label for f in self.findings)),
            "total_findings": len(self.findings),
            "findings": [
                {
                    "group": f.group_label,
                    "table_a": f.table_a,
                    "table_b": f.table_b,
                    "mean_a": f.mean_a,
                    "mean_b": f.mean_b,
                    "sd_a": f.sd_a,
                    "sd_b": f.sd_b,
                    "z_score": f.z_score,
                    "p_value": f.p_value,
                    "severity": f.severity,
                    "description": f.description,
                }
                for f in self.findings
            ],
        }


def to_markdown(result: Dict) -> str:
    """输出 Markdown 格式报告"""
    lines = [
        "# 同篇多表数据联动检查",
        "",
        f"**提取组数:** {result['total_groups']}",
        f"**冲突组数:** {result['groups_with_conflicts']}",
        f"**发现:** {result['total_findings']}",
        "",
    ]

    if not result["findings"]:
        lines.append("✅ 未发现同组跨表数据不一致。")
        return "\n".join(lines)

    lines.append("## 跨表冲突详情")
    lines.append("")
    for f in result["findings"]:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(f["severity"], "•")
        lines.append(
            f"- {icon} [{f['severity']}] {f['group']}: "
            f"{f['table_a']}({f['mean_a']}±{f['sd_a']}) vs "
            f"{f['table_b']}({f['mean_b']}±{f['sd_b']}), "
            f"z={f['z_score']:.2f}, p={f['p_value']:.4f}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="同篇论文多表数据联动检查")
    parser.add_argument("pdf", help="论文PDF路径")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", "-o", help="输出文件路径")
    args = parser.parse_args()

    checker = CrossTableConsistencyChecker()
    checker.extract_from_pdf(args.pdf)
    checker.check_consistency()
    result = checker.to_dict()

    if args.format == "markdown":
        output = to_markdown(result)
    else:
        output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"✅ 报告已保存: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
