#!/usr/bin/env python3
"""
统计方法审计器 — 三层次中文论文统计检查
==========================================

定位：从 PDF 文本中自动提取统计声明 → 判断方法适用性 → 反算验证 → 结论一致性检查。

三层架构：
┌─────────────────────────────────────────────────────────────┐
│ 第1层 方法是否正确 │ 提取声称的统计方法 → 判断是否匹配数据类型 │
│                    │ 检查方法前提（正态性/方差齐性/n值下限）   │
├─────────────────────────────────────────────────────────────┤
│ 第2层 数据是否正确 │ 用声称的方法反算 P值/F值/χ²             │  
│                    │ 对比论文报告的结果 vs 反算结果            │
├─────────────────────────────────────────────────────────────┤
│ 第3层 结论是否正确 │ 基于反算的真实统计结果                    │
│                    │ 判断论文下的结论是否成立                  │
└─────────────────────────────────────────────────────────────┘

支持的统计方法：
- 独立样本 t 检验 (Student's t-test)
- 配对 t 检验 (Paired t-test)
- 单因素方差分析 (One-way ANOVA)
- 重复测量方差分析 (Repeated Measures ANOVA)
- 卡方检验 (Chi-square test)
- Fisher 精确检验 (Fisher's exact test)
- Mann-Whitney U 检验（非参数）
- Wilcoxon 符号秩检验（非参数）
- Kruskal-Wallis 检验（非参数）
- Pearson 相关分析
- Spearman 等级相关

用法:
    python3 stat_method_auditor.py 论文.pdf [--output report.json]

依赖: PyMuPDF (fitz), scipy, numpy
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


@dataclass
class StatDeclaration:
    """论文中的统计声明"""
    method: str            # 声称的统计方法名
    raw_text: str          # 原文片段
    line_number: int       # 行号
    software: Optional[str] = None  # SPSS/SAS/GraphPad 等
    p_threshold: Optional[float] = None  # P < 0.05 等

@dataclass  
class AuditFinding:
    """审计发现"""
    layer: int             # 1/2/3
    severity: str          # HIGH/MEDIUM/LOW/INFO
    category: str          # 发现类别
    description: str       # 人类可读描述
    expected: Optional[str] = None
    actual: Optional[str] = None
    statistics: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 第1层：方法适用性判断
# ═══════════════════════════════════════════════════════════════

class MethodSuitabilityChecker:
    """检查声称的统计方法是否匹配数据类型"""

    # 数据类型 → 适用方法映射
    TYPE_METHOD_MAP = {
        "continuous_normal_two_groups": {
            "recommended": ["独立样本t检验", "Student's t-test", "Welch's t-test", "Mann-Whitney U检验"],
            "acceptable": ["单因素方差分析"],
            "wrong": ["卡方检验", "χ²检验", "Fisher精确检验"],
        },
        "continuous_normal_multi_groups": {
            "recommended": ["单因素方差分析", "ANOVA", "重复测量方差分析", "Kruskal-Wallis检验"],
            "acceptable": ["独立样本t检验"],  # 事后两两比较
            "wrong": ["卡方检验"],
        },
        "continuous_paired": {
            "recommended": ["配对t检验", "Wilcoxon符号秩检验"],
            "acceptable": ["独立样本t检验"],  # 保守但可用
            "wrong": ["卡方检验", "单因素方差分析"],
        },
        "categorical_2x2": {
            "recommended": ["卡方检验", "χ²检验", "Fisher精确检验"],
            "acceptable": [],
            "wrong": ["t检验", "t test", "方差分析", "ANOVA"],
        },
        "categorical_rxc": {
            "recommended": ["卡方检验", "χ²检验", "Fisher精确检验"],
            "acceptable": [],
            "wrong": ["t检验", "方差分析"],
        },
        "rank_ordinal": {
            "recommended": ["Mann-Whitney U检验", "Kruskal-Wallis检验", "Wilcoxon符号秩检验", "Spearman等级相关"],
            "acceptable": [],
            "wrong": ["t检验", "方差分析", "Pearson相关"],
        },
        "survival_time": {
            "recommended": ["Kaplan-Meier", "log-rank检验", "Cox回归"],
            "acceptable": [],
            "wrong": ["t检验", "卡方检验"],
        },
        "correlation": {
            "recommended": ["Pearson相关", "Spearman等级相关"],
            "acceptable": ["线性回归"],
            "wrong": ["t检验", "卡方检验"],
        },
    }

    # 中文方法名 → 英文标准化
    METHOD_ALIASES = {
        # t检验家族
        "t检验": "独立样本t检验",
        "t test": "独立样本t检验",
        "student's t test": "独立样本t检验",
        "student t检验": "独立样本t检验",
        "welch's t test": "Welch's t-test",
        "welch t检验": "Welch's t-test",
        "配对t检验": "配对t检验",
        "paired t test": "配对t检验",
        # ANOVA家族
        "单因素方差分析": "单因素方差分析",
        "单因素分差分析": "单因素方差分析",  # 常见笔误
        "one-way anova": "单因素方差分析",
        "anova": "单因素方差分析",
        "方差分析": "单因素方差分析",
        "重复测量方差分析": "重复测量方差分析",
        "repeated measures anova": "重复测量方差分析",
        "双因素方差分析": "双因素方差分析",
        "two-way anova": "双因素方差分析",
        # 非参数
        "秩和检验": "Mann-Whitney U检验",
        "mann-whitney": "Mann-Whitney U检验",
        "mann whitney u": "Mann-Whitney U检验",
        "wilcoxon": "Wilcoxon符号秩检验",
        "wilcoxon符号秩检验": "Wilcoxon符号秩检验",
        "kruskal-wallis": "Kruskal-Wallis检验",
        "kruskal wallis": "Kruskal-Wallis检验",
        # 卡方
        "卡方检验": "卡方检验",
        "χ²检验": "卡方检验",
        "χ2检验": "卡方检验",
        "chi-square": "卡方检验",
        "chi square": "卡方检验",
        "fisher精确检验": "Fisher精确检验",
        "fisher exact": "Fisher精确检验",
        "fisher's exact": "Fisher精确检验",
        # 相关
        "pearson相关": "Pearson相关",
        "pearson correlation": "Pearson相关",
        "spearman等级相关": "Spearman等级相关",
        "spearman correlation": "Spearman等级相关",
        "spearman": "Spearman等级相关",
        # 软件
        "spss": "SPSS",
        "sas": "SAS",
        "stata": "Stata",
        "graphpad": "GraphPad",
        "graphpad prism": "GraphPad Prism",
        "prism": "GraphPad Prism",
    }

    @classmethod
    def normalize_method(cls, raw_text: str) -> Optional[str]:
        """标准化方法名"""
        t = raw_text.lower().strip()
        for alias, standard in cls.METHOD_ALIASES.items():
            if alias in t:
                return standard
        return None

    @classmethod
    def check_suitability(cls, data_type: str, declared_methods: List[str], 
                          n_values: List[int] = None) -> List[AuditFinding]:
        """检查方法适用性"""
        findings = []
        if data_type not in cls.TYPE_METHOD_MAP:
            return findings

        rules = cls.TYPE_METHOD_MAP[data_type]
        for method in declared_methods:
            normalized = cls.normalize_method(method)
            if not normalized:
                continue

            if normalized in rules["wrong"]:
                findings.append(AuditFinding(
                    layer=1, severity="HIGH",
                    category="method_mismatch",
                    description=f"声称的统计方法 '{method}' ({normalized}) 不适用于 {data_type} 类型数据",
                    expected=", ".join(rules["recommended"]),
                    actual=method,
                ))
            elif normalized in rules["acceptable"]:
                findings.append(AuditFinding(
                    layer=1, severity="LOW",
                    category="method_suboptimal",
                    description=f"'{method}' ({normalized}) 可用但非最优，推荐 {rules['recommended'][0]}",
                ))

        # n值检查：t检验要求每组≥3
        if n_values and any("t检验" in m or "t test" in m.lower() for m in declared_methods):
            small_n = [n for n in n_values if n < 3]
            if small_n:
                findings.append(AuditFinding(
                    layer=1, severity="HIGH",
                    category="insufficient_sample",
                    description=f"t检验要求每组至少3个样本，但发现 n={min(small_n)}",
                ))

        return findings


# ═══════════════════════════════════════════════════════════════
# 第2层：反算验证
# ═══════════════════════════════════════════════════════════════

class StatRecalculation:
    """用论文声称的方法反算统计结果"""

    @staticmethod
    def anova_oneway(groups_data: List[Tuple[float, float, int]]) -> AuditFinding:
        """
        单因素ANOVA反算
        groups_data: [(mean, sd, n), ...]
        返回: 比较论文P值 vs 反算P值
        """
        means = [g[0] for g in groups_data]
        sds = [g[1] for g in groups_data]
        ns = [g[2] for g in groups_data]

        k = len(groups_data)
        N = sum(ns)

        grand_mean = sum(m * n for m, n in zip(means, ns)) / N
        ss_between = sum(n * (m - grand_mean)**2 for m, n in zip(means, ns))
        ss_within = sum((n - 1) * sd**2 for sd, n in zip(sds, ns))

        if ss_within == 0:
            f_stat = float('inf')
            p_value = 0.0
        else:
            f_stat = (ss_between / (k - 1)) / (ss_within / (N - k))
            p_value = 1 - stats.f.cdf(f_stat, k - 1, N - k)

        return AuditFinding(
            layer=2, severity="INFO",
            category="anova_recalculation",
            description=f"ANOVA反算: F({k-1},{N-k})={f_stat:.2f}, P={p_value:.4e}",
            statistics={
                "method": "one-way ANOVA",
                "F_statistic": f_stat,
                "df1": k - 1, "df2": N - k,
                "p_recalculated": p_value,
                "groups": [{"mean": m, "sd": s, "n": n} for m, s, n in groups_data],
            }
        )

    @staticmethod
    def t_test_independent(m1: float, sd1: float, n1: int,
                           m2: float, sd2: float, n2: int) -> AuditFinding:
        """独立样本t检验反算"""
        # 合并方差 (pooled variance, 假设方差齐性)
        sp2 = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2)
        se = math.sqrt(sp2 * (1/n1 + 1/n2))
        t_stat = (m1 - m2) / se if se > 0 else 0
        df = n1 + n2 - 2
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return AuditFinding(
            layer=2, severity="INFO",
            category="t_test_recalculation",
            description=f"独立样本t检验反算: t({df})={t_stat:.2f}, P={p_value:.4f}",
            statistics={
                "method": "independent t-test (pooled variance)",
                "t_statistic": t_stat, "df": df, "p_recalculated": p_value,
                "group1": {"mean": m1, "sd": sd1, "n": n1},
                "group2": {"mean": m2, "sd": sd2, "n": n2},
            }
        )

    @staticmethod
    def chi_square(observed: List[List[int]]) -> AuditFinding:
        """卡方检验反算"""
        obs = np.array(observed)
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(obs, correction=False)
        except ValueError:
            return AuditFinding(
                layer=2, severity="ERROR",
                category="chi_square_failed",
                description="卡方检验反算失败（数据不足或预期频数过小）",
            )

        need_fisher = np.any(expected < 5)
        return AuditFinding(
            layer=2, severity="MEDIUM" if need_fisher else "INFO",
            category="chi_square_recalculation",
            description=(
                f"卡方检验反算: χ²({dof})={chi2:.2f}, P={p_value:.4f}"
                + (" ⚠️ 预期频数<5，应使用Fisher精确检验" if need_fisher else "")
            ),
            statistics={
                "method": "chi-square",
                "chi2": chi2, "df": dof, "p_recalculated": p_value,
                "needs_fisher": need_fisher,
            }
        )


# ═══════════════════════════════════════════════════════════════
# 第3层：结论一致性检查
# ═══════════════════════════════════════════════════════════════

def check_conclusion_consistency(text: str, recalculated_findings: List[AuditFinding]) -> List[AuditFinding]:
    """
    检查论文结论是否与反算的统计结果一致
    """
    findings = []

    # 提取论文中声称"差异有统计学意义"的声明
    sig_patterns = [
        (r"差异[有具]统计学意义.*?P\s*[<＜]\s*0\.\d+", "显著声明 → 需要反算P<阈值"),
        (r"差异无统计学意义.*?P\s*[>＞]\s*0\.\d+", "不显著声明 → 需要反算P>阈值"),
        (r"明显[高低升降增减].*?P\s*[<＜]\s*0\.\d+", "效果声明"),
    ]

    for pattern, desc in sig_patterns:
        matches = re.findall(pattern, text)
        for m in matches[:10]:
            findings.append(AuditFinding(
                layer=3, severity="INFO",
                category="conclusion_statement",
                description=f"论文声明: {m[:100]}",
                statistics={"pattern": desc},
            ))

    # P值语法矛盾
    contradictions = []
    contradictions += re.findall(r"差异无统计学意义.*?P\s*[<＜]\s*0\.\d+", text)
    contradictions += re.findall(r"差异[有具]统计学意义.*?P\s*[>＞]\s*0\.\d+", text)
    for c in contradictions[:5]:
        findings.append(AuditFinding(
            layer=3, severity="HIGH",
            category="p_value_contradiction",
            description=f"P值语法矛盾: 文字描述与P值方向不一致 — '{c[:80]}'",
        ))

    # 检查"各时间点均显著"类全称量化声明
    all_timepoints = re.findall(r"各时间点[均都]?.*?P\s*[<＜]\s*0\.\d+", text)
    for at in all_timepoints[:5]:
        findings.append(AuditFinding(
            layer=3, severity="MEDIUM",
            category="sweeping_claim",
            description=f"全称量化声明: '{at[:80]}' — 需逐时间点验证",
        ))

    return findings


# ═══════════════════════════════════════════════════════════════
# 主审计流程
# ═══════════════════════════════════════════════════════════════

class StatMethodAuditor:
    """三层次统计方法审计器"""

    def __init__(self):
        self.findings: List[AuditFinding] = []
        self.raw_text: str = ""
        self.declarations: List[StatDeclaration] = []

    def load_pdf(self, pdf_path: str) -> str:
        """加载PDF文本"""
        if not HAS_PYMUPDF:
            raise ImportError("需要 PyMuPDF: pip install PyMuPDF")
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        self.raw_text = "\n".join(text_parts)
        return self.raw_text

    def extract_declarations(self) -> List[StatDeclaration]:
        """从文本中提取所有统计声明"""
        if not self.raw_text:
            return []

        lines = self.raw_text.split("\n")
        declarations = []

        # 统计方法声明模式 — 更宽松的匹配，捕获各种中文表达
        method_patterns = [
            # 模式1: 各种"用XX方法"的表达
            r"(?:采用|使用|应用|以|用|经|行)\s*(?:SPSS\s*[\d.]+|SAS|Stata|GraphPad\s*Prism\s*[\d.]+|Prism\s*[\d.]+)?\s*[^\s。，]{0,20}?"
            r"((?:独立样本\s*)?(?:配对\s*)?t\s*(?:检验|test)|单因素分差分析|单因素方差分析|"
            r"双因素方差分析|重复测量方差分析|方差分析|ANOVA|"
            r"卡方检验|χ[²2]\s*检验|chi-square|"
            r"秩和检验|Mann[-\s]Whitney\s*U|Wilcoxon|Kruskal[-\s]Wallis|"
            r"Fisher\s*精确检验|log[-\s]rank|Kaplan[-\s]Meier|Cox\s*(?:回归|模型)|"
            r"Pearson\s*(?:相关|相关分析|相关系数)|Spearman\s*(?:相关|等级相关|等级相关分析)|"
            r"q\s*检验|LSD[-\s]t\s*检验|SNK\s*检验|Tukey\s*检验|Bonferroni)",
            
            # 模式2: "各组比较采用XX" 类
            r"(?:各组|组间|两组|多组)[^。]{0,20}(?:比较|分析|采用|用)\s*"
            r"((?:独立样本\s*)?(?:配对\s*)?t\s*(?:检验|test)|单因素分差分析|单因素方差分析|"
            r"方差分析|χ[²2]检验|卡方检验|秩和检验|q检验)",
            
            # 模式3: x̄±s 声明
            r"((?:实验数据|计量资料|数据|结果)?以\s*(?:珋)?\s*x̄\s*[±±±]\s*s\s*(?:表示|来表达))",
            
            # 模式4: P值阈值
            r"(?:以|取)\s*P\s*[<＜]\s*(0\.\d+)\s*(?:为|作为)\s*(?:差异)?有?统计学意义",
            
            # 模式5: 事后比较方法
            r"((?:LSD|SNK|Tukey|Bonferroni|Dunnett|Student-Newman-Keuls|q\s*检验|Newman[-\s]Keuls)\s*(?:法|检验|比较)?)",
            
            # 模式6: 软件声明
            r"((?:SPSS|SAS|Stata|GraphPad\s*Prism|Prism)\s*[\d.]*\s*(?:软件|统计软件|统计学软件))",
        ]

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue

            for pattern in method_patterns:
                for m in re.finditer(pattern, stripped, re.IGNORECASE):
                    declarations.append(StatDeclaration(
                        method=m.group(0),
                        raw_text=stripped,
                        line_number=i + 1,
                    ))

        # 软件检测
        software_kw = ["SPSS", "SAS", "Stata", "GraphPad", "Prism", "R软件", "R语言", "Excel"]
        for kw in software_kw:
            if kw.lower() in self.raw_text.lower():
                # 找到第一个匹配行
                for i, line in enumerate(lines):
                    if kw.lower() in line.lower():
                        declarations.append(StatDeclaration(
                            method=f"软件: {kw}",
                            raw_text=line.strip(),
                            line_number=i + 1,
                            software=kw,
                        ))
                        break

        # P值阈值
        p_match = re.search(r"P\s*[<＜]\s*(0\.\d+)", self.raw_text)
        if p_match:
            declarations.append(StatDeclaration(
                method=f"P值阈值: P<{p_match.group(1)}",
                raw_text=p_match.group(0),
                line_number=0,
                p_threshold=float(p_match.group(1)),
            ))

        self.declarations = declarations
        return declarations

    def audit_layer1(self, data_type: str = "unknown") -> List[AuditFinding]:
        """第1层：方法是否正确"""
        methods = [d.method for d in self.declarations]

        # 提取n值
        n_matches = re.findall(r"[nN]\s*[=＝]\s*(\d+)", self.raw_text)
        n_values = [int(n) for n in n_matches] if n_matches else None

        findings = MethodSuitabilityChecker.check_suitability(
            data_type, methods, n_values
        )

        # 方法前提检查
        if any("t检验" in m or "t test" in m.lower() for m in methods):
            if not any("正态" in self.raw_text for _ in [1]):
                findings.append(AuditFinding(
                    layer=1, severity="MEDIUM",
                    category="missing_normality_check",
                    description="使用t检验但未提及正态性检验——应报告是否检验了数据正态分布",
                ))

        if any("方差分析" in m or "anova" in m.lower() for m in methods):
            if not any(kw in self.raw_text.lower() for kw in ["方差齐性", "levene", "bartlett"]):
                findings.append(AuditFinding(
                    layer=1, severity="MEDIUM",
                    category="missing_homoscedasticity_check",
                    description="使用方差分析但未提及方差齐性检验（Levene/Bartlett）",
                ))

        self.findings.extend(findings)
        return findings

    def audit_layer2(self, data_blocks: List[Dict] = None) -> List[AuditFinding]:
        """
        第2层：反算验证
        data_blocks: [{"type": "anova|t_test|chi_square", "data": ...}, ...]
        """
        findings = []
        if not data_blocks:
            return findings

        for block in data_blocks:
            dtype = block.get("type", "")
            data = block.get("data", {})
            reported_p = block.get("reported_p", None)

            recalc = None
            if dtype == "anova":
                recalc = StatRecalculation.anova_oneway(data.get("groups", []))
            elif dtype == "t_test":
                recalc = StatRecalculation.t_test_independent(**data)
            elif dtype == "chi_square":
                recalc = StatRecalculation.chi_square(data.get("observed", []))

            if recalc:
                # 如果论文报告了P值，比对
                if reported_p is not None and "p_recalculated" in recalc.statistics:
                    p_recalc = recalc.statistics["p_recalculated"]
                    if p_recalc > 0.05 and reported_p < 0.05:
                        recalc.severity = "HIGH"
                        recalc.description += (
                            f" ❌ 论文报告P<0.05但反算P={p_recalc:.4f}（不显著）！"
                        )
                findings.append(recalc)

        self.findings.extend(findings)
        return findings

    def audit_layer3(self) -> List[AuditFinding]:
        """第3层：结论一致性检查"""
        findings = check_conclusion_consistency(self.raw_text, self.findings)
        self.findings.extend(findings)
        return findings

    def audit(self, pdf_path: str, data_type: str = "unknown",
              data_blocks: List[Dict] = None) -> Dict:
        """
        完整三层次审计
        返回 JSON 格式报告
        """
        self.findings = []
        self.load_pdf(pdf_path)
        self.extract_declarations()

        layer1 = self.audit_layer1(data_type)
        layer2 = self.audit_layer2(data_blocks)
        layer3 = self.audit_layer3()

        # 汇总
        high_count = sum(1 for f in self.findings if f.severity == "HIGH")
        med_count = sum(1 for f in self.findings if f.severity == "MEDIUM")

        if high_count >= 3:
            risk = "RED severe"
        elif high_count >= 1 or med_count >= 5:
            risk = "ORANGE high"
        elif med_count >= 1:
            risk = "YELLOW moderate"
        else:
            risk = "GREEN low"

        return {
            "tool": "stat_method_auditor",
            "input": pdf_path,
            "risk_level": risk,
            "declarations": [
                {"method": d.method, "line": d.line_number, "software": d.software}
                for d in self.declarations
            ],
            "findings": [
                {
                    "layer": f.layer,
                    "severity": f.severity,
                    "category": f.category,
                    "description": f.description,
                    "expected": f.expected,
                    "actual": f.actual,
                    "statistics": f.statistics,
                }
                for f in self.findings
            ],
            "summary": {
                "total_declarations": len(self.declarations),
                "total_findings": len(self.findings),
                "high": high_count,
                "medium": med_count,
            },
        }


def to_markdown(result: Dict) -> str:
    lines = [
        "# 三层次统计方法审计报告",
        "",
        f"**风险等级:** {result['risk_level']}",
        f"**统计声明数:** {result['summary']['total_declarations']}",
        f"**发现:** HIGH={result['summary']['high']}, MEDIUM={result['summary']['medium']}",
        "",
        "## 统计声明提取",
        "",
    ]
    for d in result.get("declarations", []):
        sw = f" ({d.get('software')})" if d.get('software') else ""
        lines.append(f"- L{d['line']:04d}: {d['method']}{sw}")

    lines.extend(["", "## 审计发现", ""])
    layer_names = {1: "方法适用性", 2: "反算验证", 3: "结论一致性"}
    for layer_id in [1, 2, 3]:
        layer_findings = [f for f in result["findings"] if f["layer"] == layer_id]
        if not layer_findings:
            continue
        lines.append(f"### 第{layer_id}层：{layer_names[layer_id]}")
        lines.append("")
        for f in layer_findings:
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️", "ERROR": "❌"}.get(f["severity"], "•")
            lines.append(f"- {icon} [{f['category']}] {f['description']}")
            if f.get("statistics"):
                for k, v in f["statistics"].items():
                    if isinstance(v, float):
                        lines.append(f"    - {k}: {v:.4f}")
                    elif not isinstance(v, (list, dict)):
                        lines.append(f"    - {k}: {v}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="三层次统计方法审计器")
    parser.add_argument("pdf", help="论文PDF路径")
    parser.add_argument("--data-type", default="unknown",
                       help="数据类型 (continuous_normal_two_groups/continuous_normal_multi_groups/categorical_2x2/categorical_rxc/survival_time/correlation)")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", "-o", help="输出文件路径")
    args = parser.parse_args()

    auditor = StatMethodAuditor()
    result = auditor.audit(args.pdf, data_type=args.data_type, data_blocks=None)

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
