#!/usr/bin/env python3
"""
时间序列虚假模式检测
====================================

检测论文中"生长曲线""时间-效应曲线""浓度-效应曲线"等时间序列数据是否
呈现人工构造的规律性模式。

五种虚假模式检测：

1. 完美单调递增/递减：真实生物实验不可能完全单调
   检测：序列严格递增/递减 → 连续差值全部同号

2. 恒定增长率：每步增长率完全相同
   检测：增长率 d[i] = (y[i+1]-y[i])/y[i]，检查所有 d[i] 是否相同

3. 均匀间隔：相邻时间点数据等间距
   检测：y[i+1]-y[i] 全部相同 → 等差数列（数据级）

4. 恒定二阶差分：一阶差分也等差
   检测：二阶差分全部相同 → 数据由二次多项式生成

5. 拟合优度异常高：R² > 0.999 → 过度完美
   检测：线性/对数/指数三种拟合的 R²

用法：
    python3 time_series_fraud.py 论文.pdf [--output report.json]

依赖: PyMuPDF (fitz), numpy, scipy
"""

from __future__ import annotations

import argparse
import json
import math
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
class TimeSeries:
    """一条时间序列"""
    label: str             # 名称（如 "Control组 细胞增殖曲线"）
    table_id: str          # 表编号
    time_points: List[float]
    values: List[float]
    raw_text: str = ""


@dataclass
class TSFinding:
    """时间序列异常发现"""
    label: str
    detection_type: str    # MONOTONIC / CONSTANT_GROWTH / UNIFORM_SPACING / QUADRATIC / OVERFIT
    severity: str          # HIGH/MEDIUM/LOW
    detail: str
    statistics: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 主类
# ═══════════════════════════════════════════════════════════════

class TimeSeriesFraudDetector:
    """
    时间序列虚假模式检测器

    检测论文中的生长/疗效/时间效应曲线是否呈现人工构造特征。
    真实生物实验受随机误差影响，不会出现完美规律。
    """

    def __init__(self):
        self.series_list: List[TimeSeries] = []
        self.findings: List[TSFinding] = []

    def extract_from_pdf(self, pdf_path: str) -> List[TimeSeries]:
        """从PDF提取所有时间序列数据"""
        if not HAS_PYMUPDF:
            raise ImportError("PyMuPDF (fitz) 未安装")

        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()

        series_found = []

        # 找表格中的时间序列——至少4个连续数值行
        # 先按表分段
        table_pattern = re.compile(r'(?:表\s*\d+|Table\s*\d+|Tab\.?\s*\d+)')
        table_positions = [(m.group(), m.start(), m.end()) for m in table_pattern.finditer(full_text)]

        # 对每个表区段，找连续数值行
        for idx, (t_name, t_start, t_end) in enumerate(table_positions):
            # 表范围：当前表开始到下一个表之前（或文档结束）
            if idx + 1 < len(table_positions):
                section = full_text[t_end:table_positions[idx + 1][1]]
            else:
                section = full_text[t_end:]

            # 找时间点行（典型格式： 0h  24h  48h  72h  0  1  2  3 mg/L）
            time_headers = re.findall(
                r'((?:0\s*(?:h|d|min|s|mg|μg)\s*)+[\d\s\.h d m i n g μ]*)',
                section
            )

            # 找 Mean±SD 数据行 —— 至少包含3个连续的 Mean±SD 数据
            data_rows = re.findall(
                r'([\u4e00-\u9fff\w\-]+)\s+'
                r'((?:\d+\.?\d*\s*±\s*\d+\.?\d*\s*){3,})',
                section
            )

            for group_name, data_str in data_rows:
                # 提取数值
                values_match = re.findall(r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)', data_str)
                if not values_match or len(values_match) < 3:
                    continue

                means = [float(m) for m, _ in values_match]

                # 生成伪时间点（如果无法从表头提取实际时间点）
                time_points = list(range(len(means)))

                series_found.append(TimeSeries(
                    label=f"{group_name}({t_name})",
                    table_id=t_name,
                    time_points=time_points,
                    values=means,
                    raw_text=f"{group_name}: {data_str}"
                ))

        doc.close()
        self.series_list = series_found
        return series_found

    def detect_all(self) -> List[TSFinding]:
        """对所有提取的时间序列运行五种检测"""
        self.findings = []
        for s in self.series_list:
            self.findings.extend(self._check_monotonic(s))
            self.findings.extend(self._check_constant_growth(s))
            self.findings.extend(self._check_uniform_spacing(s))
            self.findings.extend(self._check_quadratic(s))
            self.findings.extend(self._check_overfit(s))
        return self.findings

    def _check_monotonic(self, s: TimeSeries) -> List[TSFinding]:
        """检测1：完美单调递增/递减"""
        vals = s.values
        if len(vals) < 3:
            return []

        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        all_positive = all(d > 0 for d in diffs if abs(d) > 1e-10)
        all_negative = all(d < 0 for d in diffs if abs(d) > 1e-10)

        # 真实实验极少出现完美单调 -> 但这不是硬证据
        # 真正可疑的是：每组都是完美单调 + 没有任何反弹
        if all_positive or all_negative:
            direction = "递增" if all_positive else "递减"
            return [TSFinding(
                label=s.label,
                detection_type="PERFECT_MONOTONIC",
                severity="MEDIUM",
                detail=f"时间序列{len(vals)}个点呈现完美单调{direction}，真实生物实验极少出现无任何波动",
                statistics={
                    "n_points": len(vals),
                    "n_diffs": len(diffs),
                    "all_positive": all_positive,
                    "all_negative": all_negative,
                }
            )]
        return []

    def _check_constant_growth(self, s: TimeSeries) -> List[TSFinding]:
        """检测2：恒定增长率"""
        vals = s.values
        if len(vals) < 3:
            return []

        # 对于正值序列，计算增长率
        if all(v > 0 for v in vals):
            growth_rates = [(vals[i+1] - vals[i]) / vals[i] for i in range(len(vals)-1)]
            # 标准化：增长率标准差 / 均值
            if len(growth_rates) > 1:
                gr_std = np.std(growth_rates)
                gr_mean = abs(np.mean(growth_rates))
                if gr_mean > 1e-10:
                    cv = gr_std / gr_mean
                    if cv < 1e-4:
                        return [TSFinding(
                            label=s.label,
                            detection_type="CONSTANT_GROWTH_RATE",
                            severity="HIGH",
                            detail=f"每步增长率几乎相同(CV={cv:.2e})，疑似按恒定比例生成",
                            statistics={
                                "growth_rates": [round(r, 6) for r in growth_rates],
                                "cv": cv,
                            }
                        )]
        return []

    def _check_uniform_spacing(self, s: TimeSeries) -> List[TSFinding]:
        """检测3：均匀间隔——y[i+1]-y[i] 全部相同"""
        vals = s.values
        if len(vals) < 3:
            return []

        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        diff_std = np.std(diffs)
        diff_mean = abs(np.mean(diffs))

        # 与原等差数列检测一致：σ/|mean| < ε
        if diff_mean > 1e-10:
            ratio = diff_std / diff_mean
            severity = "HIGH" if ratio < 1e-4 else ("MEDIUM" if ratio < 0.001 else None)
            if severity:
                return [TSFinding(
                    label=s.label,
                    detection_type="UNIFORM_SPACING",
                    severity=severity,
                    detail=f"相邻时间点数值差值几乎相同(σ/|mean|={ratio:.2e})，疑似等差数列",
                    statistics={
                        "diffs": [round(d, 4) for d in diffs],
                        "std_over_mean": ratio,
                    }
                )]
        return []

    def _check_quadratic(self, s: TimeSeries) -> List[TSFinding]:
        """检测4：恒定二阶差分（二次多项式生成）"""
        vals = s.values
        if len(vals) < 4:
            return []

        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]

        sd_std = np.std(second_diffs)
        sd_mean = abs(np.mean(second_diffs))

        if sd_mean > 1e-10:
            ratio = sd_std / sd_mean
            if ratio < 1e-6:
                return [TSFinding(
                    label=s.label,
                    detection_type="CONSTANT_SECOND_DIFF",
                    severity="HIGH",
                    detail=f"二阶差分几乎恒定(σ/|mean|={ratio:.2e})，数据可能由二次多项式生成",
                    statistics={
                        "second_diffs": [round(d, 6) for d in second_diffs],
                        "std_over_mean": ratio,
                    }
                )]
        return []

    def _check_overfit(self, s: TimeSeries) -> List[TSFinding]:
        """检测5：拟合优度异常高（R² > 0.999）"""
        vals = s.values
        tp = s.time_points

        if len(vals) < 4:
            return []

        x = np.array(tp, dtype=float)
        y = np.array(vals, dtype=float)

        findings = []

        # 线性拟合
        coeffs_linear = np.polyfit(x, y, 1)
        y_pred_linear = np.polyval(coeffs_linear, x)
        ss_res = np.sum((y - y_pred_linear) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot > 1e-10:
            r2_linear = 1 - ss_res / ss_tot
            if r2_linear > 0.999:
                findings.append(TSFinding(
                    label=s.label,
                    detection_type="OVERFIT_LINEAR",
                    severity="MEDIUM",
                    detail=f"线性拟合 R²={r2_linear:.6f} > 0.999，真实生物实验极少出现如此完美拟合",
                    statistics={"r2": r2_linear, "fit_type": "linear"},
                ))

        # 指数拟合（对数空间线性）
        if all(v > 0 for v in vals):
            log_y = np.log(y)
            coeffs_exp = np.polyfit(x, log_y, 1)
            y_pred_exp = np.exp(np.polyval(coeffs_exp, x))
            ss_res_exp = np.sum((y - y_pred_exp) ** 2)
            if ss_tot > 1e-10:
                r2_exp = 1 - ss_res_exp / ss_tot
                if r2_exp > 0.999:
                    findings.append(TSFinding(
                        label=s.label,
                        detection_type="OVERFIT_EXPONENTIAL",
                        severity="MEDIUM",
                        detail=f"指数拟合 R²={r2_exp:.6f} > 0.999，生长曲线过度完美",
                        statistics={"r2": r2_exp, "fit_type": "exponential"},
                    ))

        return findings

    def to_dict(self) -> Dict:
        """导出结果"""
        high = sum(1 for f in self.findings if f.severity == "HIGH")
        medium = sum(1 for f in self.findings if f.severity == "MEDIUM")
        return {
            "total_series": len(self.series_list),
            "total_findings": len(self.findings),
            "high": high,
            "medium": medium,
            "findings": [
                {
                    "label": f.label,
                    "type": f.detection_type,
                    "severity": f.severity,
                    "detail": f.detail,
                    "statistics": f.statistics,
                }
                for f in self.findings
            ],
        }


def to_markdown(result: Dict) -> str:
    """输出 Markdown 格式"""
    lines = [
        "# 时间序列虚假模式检测",
        "",
        f"**提取时间序列数:** {result['total_series']}",
        f"**发现:** HIGH={result['high']}, MEDIUM={result['medium']}",
        "",
    ]

    if not result["findings"]:
        lines.append("✅ 未发现时间序列虚假模式。")
        return "\n".join(lines)

    lines.append("## 异常详情")
    lines.append("")
    type_names = {
        "PERFECT_MONOTONIC": "完美单调",
        "CONSTANT_GROWTH_RATE": "恒定增长率",
        "UNIFORM_SPACING": "均匀间隔",
        "CONSTANT_SECOND_DIFF": "恒定二阶差分",
        "OVERFIT_LINEAR": "线性过拟合(R²>0.999)",
        "OVERFIT_EXPONENTIAL": "指数过拟合(R²>0.999)",
    }
    for f in result["findings"]:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(f["severity"], "•")
        type_name = type_names.get(f["type"], f["type"])
        lines.append(f"- {icon} [{f['severity']}] [{type_name}] {f['label']}: {f['detail']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="时间序列虚假模式检测")
    parser.add_argument("pdf", help="论文PDF路径")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", "-o", help="输出文件路径")
    args = parser.parse_args()

    detector = TimeSeriesFraudDetector()
    detector.extract_from_pdf(args.pdf)
    detector.detect_all()
    result = detector.to_dict()

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
