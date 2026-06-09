#!/usr/bin/env python3
"""
Heaven's Net — 学术不端检测系统 v1.2
=======================================
天网恢恢，疏而不漏。
中英文双轨学术论文造假检测 CLI。

用法:
    heavens-net <论文目录或PDF路径> [选项]

示例:
    # 审查单篇论文
    heavens-net paper.pdf

    # 审查整个目录
    heavens-net ~/论文目录/

    # 指定输出格式
    heavens-net ~/论文目录/ --output report.json
    heavens-net ~/论文目录/ --output report.md --format markdown

    # 跨文件模式（检测跨论文的数据复制）
    heavens-net ~/论文目录/ --cross-file

输入:
    - 单个 PDF 文件路径（中英文均可）
    - 包含多个 PDF 的目录路径

输出:
    - JSON: 结构化报告，包含每篇论文的完整检测结果
    - Markdown: 人类可读报告
    - 默认: 终端输出摘要 + JSON 文件

流程（逐篇管线化过筛）:
    Step ① 身份确认 — 提取作者/通讯/单位/期刊/基金号
    Step ② 数据层判断 — 判断数据类型，注册武器适用性
    Step ③ 逐段读方法 — 提取统计声明/P值/n值
    Step ③½ 跨表数据联动 — 同篇论文不同表格的同一组数据是否一致
    Step ④ 第1层全量 — 统计反算（十二武器）
    Step ④½ 时间序列检测 — 生长曲线/时间效应虚假模式
    Step ⑤ 第2层全量 — 报告格式+算术模式
    Step ⑥ 第3层全量 — 图片取证+文本异常
    Step ⑦ 第4+5层全量 — 跨论文归因+外部验证
    Step ⑧ 独立报告

依赖:
    - PyMuPDF (fitz) — PDF 文本和图片提取
    - arithmetic_sequence_detector.py — 算术序列检测
    - image_forensics_pipeline.py — ELA + 块重复 + 背景突变
    - analyze_wb_bands.py — Western Blot 条带互相关
    - bulk_cross_paper_analysis.py — 跨论文 pHash + SSIM 比对
    - sift_duplicate_detection.py — SIFT 特征点匹配（旋转/翻转不变）
    - chinese_tortured_phrases.py — 中文文本异常
    - twelve_weapons.py — 统计反算 12 武器
    - cross_paper_email_analyzer.py — 跨论文邮箱分析
    - author_anomaly_detector.py — 作者异常检测
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ----- 路径配置 -----
# 独立发行版：所有脚本在同级 scripts/ 目录
SCRIPT_DIR = Path(__file__).parent

# 将 scripts 目录加入 sys.path（所有检测脚本都在同一目录）
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ----- 导入各检测武器 -----
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("⚠️ PyMuPDF 未安装，PDF 文本提取功能不可用。pip install PyMuPDF")

try:
    from arithmetic_sequence_detector import ArithmeticSequenceDetector
    HAS_ARITHMETIC = True
except ImportError:
    HAS_ARITHMETIC = False
    print("⚠️ 算术检测器不可用")

try:
    from twelve_weapons import TwelveWeapons
    HAS_TWELVE = True
except ImportError:
    HAS_TWELVE = False

try:
    from chinese_tortured_phrases import ChineseTorturedPhrases
    HAS_TORTURED = True
except ImportError:
    HAS_TORTURED = False

try:
    from cross_paper_email_analyzer import CrossPaperEmailAnalyzer
    HAS_EMAIL = True
except ImportError:
    HAS_EMAIL = False

try:
    from stat_method_auditor import StatMethodAuditor
    HAS_STAT_AUDITOR = True
except ImportError:
    HAS_STAT_AUDITOR = False

try:
    from author_anomaly_detector import AuthorAnomalyDetector
    HAS_AUTHOR = True
except ImportError:
    HAS_AUTHOR = False

try:
    from cross_table_consistency import CrossTableConsistencyChecker
    HAS_CROSS_TABLE = True
except ImportError:
    HAS_CROSS_TABLE = False

try:
    from time_series_fraud import TimeSeriesFraudDetector
    HAS_TIME_SERIES = True
except ImportError:
    HAS_TIME_SERIES = False

# ----- 图片取证模块（来自公开方法） -----
try:
    from image_forensics_pipeline import analyze_figure, extract_figures, cross_compare
    HAS_IMAGE_FORENSICS = True
except ImportError:
    HAS_IMAGE_FORENSICS = False

try:
    from analyze_wb_bands import analyze_wb
    HAS_WB = True
except ImportError:
    HAS_WB = False

try:
    from bulk_cross_paper_analysis import cross_paper_similarity, compute_phash
    HAS_CROSS_PAPER_IMG = True
except ImportError:
    HAS_CROSS_PAPER_IMG = False

try:
    from sift_duplicate_detection import analyze_pair
    HAS_SIFT = True
except ImportError:
    HAS_SIFT = False


# ==================================================================
# Step ①: 身份确认
# ==================================================================
def step1_identity(pdf_path: str) -> dict:
    """提取论文元数据：作者/通讯/单位/期刊/基金号（中英文双路径）"""
    result = {
        "file": str(pdf_path),
        "filename": Path(pdf_path).name,
        "authors": [],
        "corresponding": [],
        "first_author": None,
        "institution": None,
        "journal": None,
        "funding": [],
        "doi": None,
        "language": "unknown",
    }
    
    if not HAS_PYMUPDF:
        result["error"] = "PyMuPDF not available"
        return result
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(min(5, len(doc))):
            page = doc[page_num]
            full_text += page.get_text()
        # 在关闭前保存第1-2页全文（用于英文作者提取）
        first_two_pages = ""
        for page_num in range(min(2, len(doc))):
            first_two_pages += doc[page_num].get_text()
        doc.close()
    except Exception as e:
        result["error"] = f"PDF read error: {e}"
        return result
    
    # 判断中英文
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', full_text[:1000]))
    result["language"] = "zh" if chinese_chars > 50 else "en"
    is_zh = result["language"] == "zh"
    
    # --- DOI ---
    doi_match = re.search(r'(?:doi|DOI)[:\s]*([0-9]+\.[0-9]+/[^\s]+)', full_text)
    if doi_match:
        result["doi"] = doi_match.group(1).rstrip('.,;')
    
    # --- 作者 ---
    if is_zh:
        # 中文：作者[：:]\s*张三，李四，王五
        cn_author = re.search(r'(?:作者[：:]\s*)([^。\n]{5,300})', full_text)
        if cn_author:
            raw = cn_author.group(1)
            # 清理数字编号和标记
            raw = re.sub(r'[\d①②③④⑤⑥⑦⑧⑨⑩*△#†‡§¶]', '', raw)
            result["authors"] = [a.strip() for a in re.split(r'[，,;；\s]+', raw) if len(a.strip()) >= 2 and not a.strip().isdigit()]
    else:
        # 英文：用第1-2页全文（摘要长的论文会把作者行推到第1页后半甚至第2页）
        first_page = first_two_pages
        first_3000 = first_page[:3000]
        
        # 策略1: 找 "Authors:" 或 "Author list" 后的文字
        en_author = re.search(r'(?:Authors?|Author\s*list)[:\s]+([^\n]{20,400})', first_3000, re.IGNORECASE)
        if en_author:
            raw = en_author.group(1)
            # 分割: 逗号分隔或数字上标分隔
            authors = re.split(r',\s*(?=[A-Z])|\d,\s*|\d\s+(?=[A-Z])', raw)
            result["authors"] = [a.strip().rstrip(',.') for a in authors if len(a.strip()) > 2]
        
        # 策略2: 匹配 "NAME1, NAME2 and NAME3" 格式（全大写字母+数字上标）
        if not result["authors"]:
            # 真正的作者行特征: 每个姓名后紧跟数字（上标编号），逗号或and分隔
            # 例: "YANYI LI1, HUIHUI KE1, RUI ZHANG1, JIANLONG ZHU1 and MINGHUA YU2"
            author_line = re.search(
                r'([A-Z]{2,}(?:\s+[A-Z]{2,})?\d{1,2}(?:,\s+|\s+and\s+)){2,}[A-Z]{2,}(?:\s+[A-Z]{2,})?\d{1,2}',
                first_page
            )
            if author_line:
                raw = author_line.group(0)
                # 去掉每个姓名后面的数字
                raw = re.sub(r'\d{1,2}', '', raw)
                # 按逗号或and分割
                parts = re.split(r',\s*|\s+and\s+', raw)
                parts = [p.strip() for p in parts if len(p.strip()) > 3]
                # title() 转首字母大写（原来是全大写）
                result["authors"] = [p.strip().title() for p in parts]
    
    # 第一作者
    if result["authors"]:
        result["first_author"] = result["authors"][0]
    
    # --- 通讯作者 ---
    if is_zh:
        corr_match = re.search(r'(?:通讯作者|通信作者)[：:\s]*([^\n]{3,50})', full_text)
        if corr_match:
            result["corresponding"] = [corr_match.group(1).strip()]
        else:
            corr_star = re.findall(r'([\u4e00-\u9fff]{2,4})[\*△](?:\s|,|，|$)', full_text[:2000])
            if corr_star:
                result["corresponding"] = list(dict.fromkeys(corr_star))
    else:
        # 英文：找 "Correspondence to:" 或 "*Corresponding author" 或标 * 的作者
        corr_en = re.search(r'(?:Correspondence\s*to|Corresponding\s*author)[:\s]*([^\n,]{5,80})', full_text, re.IGNORECASE)
        if corr_en:
            name = corr_en.group(1).strip()
            # 清理邮箱
            name = re.sub(r'\S+@\S+', '', name).strip()
            if name:
                result["corresponding"] = [name]
        else:
            # 找标 * 的作者，但排除 PMID/DOI 行中的数字加星号
            # 只看前2000字符（标题页区域）
            title_area = full_text[:2000]
            corr_star = re.findall(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})[\*△#](?:\s|,|\.|$)', title_area)
            if corr_star:
                # 过滤掉明显的非人名（全大写、含数字、太短）
                filtered = [c for c in corr_star if len(c) > 3 and not c.isupper() and not re.search(r'\d', c)]
                if filtered:
                    result["corresponding"] = list(dict.fromkeys(filtered))
    
    # --- 单位 ---
    if is_zh:
        inst_match = re.search(r'(?:单位|机构)[：:\s]*([^\n]{5,200})', full_text)
    else:
        inst_match = re.search(r'(?:Department|Institute|College|University|Hospital|School)[^\n]{5,200}', full_text[:3000])
    if inst_match:
        result["institution"] = inst_match.group(0).strip()[:200]
    
    # --- 期刊 ---
    if not is_zh:
        # 从全文第一行或 PDF 元数据中找期刊名
        # 常见格式: "ONCOLOGY LETTERS 22: 13001, 2021" 或第一页页眉
        header_match = re.search(r'([A-Z][A-Z\s&]+)\s+\d+[:\s]', full_text[:500])
        if header_match:
            result["journal"] = header_match.group(1).strip()
        else:
            journal_match = re.search(r'(?:ONCOLOGY|CANCER|NATURE|SCIENCE|CELL|LANCET|BMJ|PLOS|JOURNAL)[A-Z\s]*', full_text[:500])
            if journal_match:
                result["journal"] = journal_match.group(0).strip()
    
    # --- 基金号 ---
    fund_keywords_zh = r'(?:基金项目|基金|资助)'
    fund_keywords_en = r'(?:Funding|Grant|Supported\s*by|Acknowledgments?)'
    fund_matches = re.findall(f'(?:{fund_keywords_zh}|{fund_keywords_en})[：:\s]*([^\n]{{5,400}})',
                              full_text, re.IGNORECASE)
    if fund_matches:
        for fm in fund_matches[:3]:
            numbers = re.findall(r'(\d{7,10})', fm)
            result["funding"].extend(numbers)
    
    result["funding"] = list(set(result["funding"]))
    
    return result


# ==================================================================
# Step ②: 数据层判断
# ==================================================================
def step2_data_layer(pdf_path: str) -> dict:
    """判断论文的数据类型，注册适用的武器"""
    result = {
        "data_type": "unknown",
        "has_mean_sd_table": False,     # 临床 Mean±SD 表
        "has_bar_chart": False,         # 柱状图
        "has_wb_images": False,         # Western Blot 图
        "has_geo_data": False,          # GEO/TCGA 公共数据
        "is_survey": False,             # 问卷/计数
        "is_review": False,             # 综述
        "is_clinical_trial": False,     # 临床试验/RCT
        "has_funding": False,           # 有基金号
        "applicable_layers": {},        # {层名: 覆盖率}
    }
    
    if not HAS_PYMUPDF:
        return result
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        
        # 统计图片数
        image_count = 0
        for page in doc:
            image_count += len(page.get_images())
        
        doc.close()
    except Exception:
        return result
    
    # 判断数据类型
    text_lower = full_text.lower()
    
    # 综述
    if any(kw in full_text[:200] for kw in ['综述', 'Review', '进展', 'Progress']):
        result["is_review"] = True
        result["data_type"] = "review"
    
    # 问卷/计数
    elif any(kw in full_text[:300] for kw in ['问卷调查', '问卷', '调查表', '量表', '计数', '例数']):
        result["is_survey"] = True
        result["data_type"] = "survey"
    
    # GEO/TCGA
    elif any(kw in text_lower for kw in ['geo ', 'gse', 'tcga', 'geo accession', 'geo数据集']):
        result["has_geo_data"] = True
        result["data_type"] = "geo_data"
    
    # WB图
    elif any(kw in text_lower for kw in ['western blot', '免疫印迹', 'western-blot', 'wb ']):
        result["has_wb_images"] = True
        if not result["data_type"] or result["data_type"] == "unknown":
            result["data_type"] = "wet_lab"
    
    # 柱状图
    if any(kw in text_lower for kw in ['bar chart', '柱状图', 'barplot']):
        result["has_bar_chart"] = True
        if result["data_type"] == "unknown":
            result["data_type"] = "bar_chart"
    
    # 临床试验
    if any(kw in text_lower[:500] for kw in ['rct', 'randomized', '随机对照', '临床试验', 'clinical trial']):
        result["is_clinical_trial"] = True
        if result["data_type"] == "unknown":
            result["data_type"] = "clinical_trial"
    
    # Mean±SD 表 — 检查是否有 x̄±s 或 Mean±SD 格式
    if re.search(r'[xXx̄\bar{X}][±±±]\s*[sS]', full_text) or re.search(r'mean\s*±\s*sd', text_lower):
        result["has_mean_sd_table"] = True
        if result["data_type"] == "unknown":
            result["data_type"] = "clinical_table"
    
    # 基金号
    if re.search(r'(\d{7,10})', full_text):
        fund_keywords = ['基金', 'funding', 'grant', '资助', 'NSFC', '国家自然科学']
        if any(kw in text_lower for kw in fund_keywords):
            result["has_funding"] = True
    
    # 确定各层武器覆盖率
    if result["data_type"] == "clinical_table":
        result["applicable_layers"] = {"layer1": 100, "layer2": 100, "layer3": 100, "layer4": 100, "layer5": 100}
    elif result["data_type"] == "bar_chart":
        result["applicable_layers"] = {"layer1": 85, "layer2": 100, "layer3": 100, "layer4": 100, "layer5": 50}
    elif result["data_type"] == "wet_lab":
        result["applicable_layers"] = {"layer1": 0, "layer2": 50, "layer3": 100, "layer4": 100, "layer5": 25}
    elif result["data_type"] == "geo_data":
        result["applicable_layers"] = {"layer1": 0, "layer2": 50, "layer3": 100, "layer4": 100, "layer5": 25}
    elif result["data_type"] == "survey":
        result["applicable_layers"] = {"layer1": 0, "layer2": 0, "layer3": 25, "layer4": 25, "layer5": 0}
    elif result["data_type"] == "review":
        result["applicable_layers"] = {"layer1": 0, "layer2": 0, "layer3": 10, "layer4": 0, "layer5": 0}
    else:
        result["applicable_layers"] = {"layer1": 50, "layer2": 50, "layer3": 50, "layer4": 50, "layer5": 50}
    
    return result


# ==================================================================
# Step ③: 逐段读方法
# ==================================================================
def step3_methods(pdf_path: str) -> dict:
    """提取方法段中的统计声明、P值阈值、n值"""
    result = {
        "stat_declarations": [],       # 统计声明
        "p_threshold": None,           # P值阈值
        "n_values": [],                # n值
        "method_violations": [],       # 方法前提违反
    }
    
    if not HAS_PYMUPDF:
        return result
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
    except Exception:
        return result
    
    # P值阈值
    p_match = re.search(r'P\s*[<≤]\s*(0\.\d+)', full_text)
    if p_match:
        result["p_threshold"] = float(p_match.group(1))
    
    # n值 — 寻找 (n=XX) 格式
    n_matches = re.findall(r'[nN]\s*[=＝]\s*(\d+)', full_text)
    result["n_values"] = [int(n) for n in n_matches][:10]
    
    # 统计声明
    stat_keywords = ['t检验', 't test', '方差分析', 'ANOVA', '卡方', 'chi-square',
                     'Mann-Whitney', 'Wilcoxon', 'Kruskal-Wallis', '正态', 'normal',
                     'x̄±s', 'mean±SD', '中位数', 'median', 'IQR']
    for kw in stat_keywords:
        if kw.lower() in full_text.lower():
            result["stat_declarations"].append(kw)
    
    # 方法前提违反检测
    # n值太小做t检验
    small_n = [n for n in result["n_values"] if n < 5]
    if small_n and any('t' in sd.lower() for sd in result["stat_declarations"]):
        result["method_violations"].append(f"n={min(small_n)}太小仍用t检验")
    
    # ======== 🆕 v1.1: 三层次统计方法审计 ========
    result["stat_audit"] = {"executed": False, "findings": [], "error": None}
    if HAS_STAT_AUDITOR:
        try:
            # 推断数据类型（从 step2 传入）
            # 默认 data_type 基于检测到的统计声明
            auditor = StatMethodAuditor()
            auditor.raw_text = full_text
            auditor.extract_declarations()
            
            # 推断数据类型
            inferred_type = "unknown"
            if any(kw in full_text.lower() for kw in ["χ²", "χ2", "卡方", "计数", "例数", "构成比", "率"]):
                inferred_type = "categorical_rxc"
            elif any(kw in full_text.lower() for kw in ["kaplan", "meier", "生存", "cox", "log.rank", "log-rank"]):
                inferred_type = "survival_time"
            elif any(kw in full_text.lower() for kw in ["pearson", "spearman", "相关", "相关系数"]):
                inferred_type = "correlation"
            elif any(kw in full_text.lower() for kw in ["配对", "前后", "自身对照", "同体"]):
                inferred_type = "continuous_paired"
            elif any(kw in full_text.lower() for kw in ["多组", "方差分析", "anova", "单因素分差"]):
                inferred_type = "continuous_normal_multi_groups"
            elif any(kw in full_text.lower() for kw in ["两组", "t检验", "t test"]):
                inferred_type = "continuous_normal_two_groups"
            
            layer1_findings = auditor.audit_layer1(data_type=inferred_type)
            layer3_findings = auditor.audit_layer3()
            
            result["stat_audit"] = {
                "executed": True,
                "inferred_data_type": inferred_type,
                "declarations": [
                    {"method": d.method, "line": d.line_number, "software": d.software}
                    for d in auditor.declarations
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
                    for f in layer1_findings + layer3_findings
                ],
                "high_count": sum(1 for f in layer1_findings + layer3_findings if f.severity == "HIGH"),
                "med_count": sum(1 for f in layer1_findings + layer3_findings if f.severity == "MEDIUM"),
            }
        except Exception as e:
            result["stat_audit"]["error"] = str(e)
    
    return result


# ==================================================================
# Step ④: 第1层全量 — 统计反算（7武器）
# ==================================================================
def step4_layer1(pdf_path: str) -> dict:
    """统计反算"""
    result = {
        "executed": False,
        "findings": [],
        "error": None
    }
    
    if not HAS_TWELVE:
        result["error"] = "twelve_weapons not available"
        return result
    
    try:
        detector = TwelveWeapons()
        # twelve_weapons 需要从 PDF 提取 Mean±SD 表
        # 这里先用 PyMuPDF 提取文本中的 Mean±SD 对
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        # 提取 Mean±SD 对（中文格式：X.XX±X.XX）
        mean_sd_pattern = re.compile(r'(\d+\.?\d*)\s*[±±±]\s*(\d+\.?\d*)')
        pairs = mean_sd_pattern.findall(full_text)
        
        if pairs:
            result["mean_sd_pairs_found"] = len(pairs)
            result["pairs_sample"] = pairs[:5]
        else:
            result["mean_sd_pairs_found"] = 0
        
        result["executed"] = True
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ==================================================================
# Step ⑤: 第2层全量 — 报告格式+算术模式
# ==================================================================
def step5_layer2(pdf_path: str, pdf_dir: str = None) -> dict:
    """报告格式与算术模式检测"""
    result = {
        "executed": False,
        "arithmetic_findings": [],
        "error": None
    }
    
    if not HAS_ARITHMETIC:
        result["error"] = "arithmetic_sequence_detector not available"
        return result
    
    try:
        detector = ArithmeticSequenceDetector()
        
        # 尝试从PDF提取CSV数据（如果有嵌入数据表）
        # 这需要 supplementary_data_fetcher 的逻辑
        # 暂时标记需要手动提取
        result["note"] = "需要将论文的 Table 1 数据手动提取为 CSV 格式后运行算术检测"
        result["csv_required"] = True
        result["executed"] = True
    except Exception as e:
        result["error"] = str(e)
    
    return result


# ==================================================================
# Step ⑥: 第3层全量 — 图片+文本
# ==================================================================
def step6_layer3(pdf_path: str, output_dir: str = None) -> dict:
    """图片取证 + 文本异常 — 完整管线：
    aHash初筛→ELA+块重复+背景突变→WB泳道拼接→WB条带互相关→pHash跨图比对→SIFT特征点"""
    import tempfile
    import shutil
    
    result = {
        "image_count": 0,
        "extracted_images": [],
        "ahash_findings": [],            # 🆕 v8.4: aHash初筛
        "ela_findings": [],
        "block_dup_findings": [],
        "background_mutation_findings": [],
        "wb_lane_findings": [],          # 🆕 v8.4: WB泳道拼接
        "wb_findings": [],
        "cross_image_findings": [],
        "sift_findings": [],
        "text_findings": [],
        "error": None
    }
    
    if not HAS_PYMUPDF:
        result["error"] = "PyMuPDF not available"
        return result
    
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num, page in enumerate(doc):
            full_text += page.get_text()
            result["image_count"] += len(page.get_images())
        
        doc.close()
    except Exception as e:
        result["error"] = f"PDF read error: {e}"
        return result
    
    # ----- 3.3 文本异常检测 -----
    if HAS_TORTURED:
        try:
            detector = ChineseTorturedPhrases()
            text_findings = detector.detect(full_text)
            result["text_findings"] = text_findings[:10]
        except Exception:
            pass
    else:
        # 简单规则：P值语法矛盾
        if re.search(r'差异无统计学意义.*P\s*[<＜]\s*0\.\d+', full_text):
            result["text_findings"].append("P值语法矛盾: '差异无统计学意义' + P<XXX")
        tortured_patterns = [
            (r'差异具有统计学意义.*?P\s*>\s*0\.\d+', "P值方向矛盾"),
            (r'统计学差异.*?P\s*>\s*0\.\d+', "统计学差异+P>0.05矛盾"),
        ]
        for pattern, desc in tortured_patterns:
            if re.search(pattern, full_text):
                result["text_findings"].append(desc)
    
    # ----- 3.0 aHash 快速初筛（🆕 v8.4）-----
    # 先提取图片到临时目录，跑 aHash 初筛标记候选重复对
    # 后续 ELA/SIFT 对 aHash 标记的候选对做精细化验证
    if result["image_count"] >= 2:
        ahash_tmp = tempfile.mkdtemp(prefix="ahash_")
        try:
            figures_dir_ahash = os.path.join(ahash_tmp, "figures")
            ahash_extracted = extract_figures(pdf_path, figures_dir_ahash, min_size=100)
            if ahash_extracted and len(ahash_extracted) >= 2:
                try:
                    from image_hash_screener import screen as ahash_screen
                    ahash_result = ahash_screen([figures_dir_ahash], threshold=4)  # v8.8.1: 从6降至4，基于阴性对照基线（PLOS ONE 16.7% FPR→目标<5%）
                    if ahash_result and ahash_result.get("findings"):
                        result["ahash_findings"] = ahash_result["findings"]
                except ImportError:
                    pass  # image_hash_screener 不可用则跳过
        finally:
            try:
                shutil.rmtree(ahash_tmp)
            except Exception:
                pass
    
    # ----- 3.1 图片取证（ELA + 块重复 + 背景突变）-----
    if HAS_IMAGE_FORENSICS and result["image_count"] > 0:
        # 创建临时输出目录
        tmp_dir = output_dir or tempfile.mkdtemp(prefix="img_forensics_")
        try:
            # 提取图片
            figures_dir = os.path.join(tmp_dir, "figures")
            extracted = extract_figures(pdf_path, figures_dir, min_size=100)
            result["extracted_images"] = extracted[:100]  # 最多记录100张
            
            if extracted:
                # 对每张图跑取证分析，收集结果用于跨图比对
                analyses = []
                for fig_path in extracted[:20]:  # 最多分析20张（避免超时）
                    try:
                        finding = analyze_figure(fig_path)
                        if finding:
                            analyses.append(finding)
                            # 分类各项发现
                            f_type = finding.get("type", "unknown")
                            if f_type == "ela_anomaly":
                                result["ela_findings"].append(finding)
                            elif f_type == "block_duplication":
                                result["block_dup_findings"].append(finding)
                            elif f_type == "background_mutation":
                                result["background_mutation_findings"].append(finding)
                            else:
                                # 通用发现——记录所有字段
                                for key in ["ela_score", "block_dup_rate", "background_mutation_z"]:
                                    if key in finding and finding[key]:
                                        result["ela_findings"].append(finding)
                                        break
                    except Exception:
                        continue
                
                # 跨图比对（同论文内），需要 analyses 列表
                if len(analyses) >= 2:
                    cross = cross_compare(analyses)
                    if cross:
                        result["cross_image_findings"] = cross[:10]
        finally:
            # 清理临时目录（如果没指定输出目录）
            if not output_dir and os.path.isdir(tmp_dir):
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
    
    # ----- 3.1b AI 生成图片频域检测（🆕 v8.8）-----
    # 复用上面提取的图片，逐张做频域分析检测 GAN/扩散模型生成指纹
    if result.get("image_count", 0) > 0:
        try:
            from ai_image_detector import AIImageDetector
            ai_detector = AIImageDetector()
            ai_results = []
            for fig_path in result.get("extracted_images", [])[:20]:
                try:
                    r = ai_detector.analyze(fig_path, verbose=False)
                    if r.get("ai_score", -1) >= 0.4:
                        r["filename"] = os.path.basename(fig_path)
                        ai_results.append(r)
                except Exception:
                    pass
            if ai_results:
                result["ai_image_findings"] = ai_results
        except ImportError:
            pass  # ai_image_detector 不可用则跳过
    
    # ----- 3.2 WB 条带分析 -----
    if HAS_WB and result["image_count"] > 0:
        try:
            # analyze_wb 需要 PDF 路径，内部提取图片
            wb_result = analyze_wb(pdf_path)
            if wb_result and "findings" in wb_result:
                result["wb_findings"] = wb_result["findings"]
        except Exception as e:
            # WB分析对非WB论文会报错（如"no images >=100px"），这是正常情况
            pass
    
    # ----- 3.2b WB 泳道拼接检测（🆕 v8.4）-----
    # 与 3.2 互补：analyze_wb 做条带级互相关，本检测做泳道级拼接+重复
    # 提取图片后跑 blot_gel_lane_audit
    if result["image_count"] > 0:
        lane_tmp = tempfile.mkdtemp(prefix="wblane_")
        try:
            figures_dir_lane = os.path.join(lane_tmp, "figures")
            lane_extracted = extract_figures(pdf_path, figures_dir_lane, min_size=100)
            if lane_extracted:
                try:
                    from blot_gel_lane_audit import audit as lane_audit
                    lane_result = lane_audit([figures_dir_lane], lanes=8, seam_z=5.0)  # v8.8.1: 从3.0升至5.0，基于阴性对照基线（PLOS ONE 73.3% FPR→目标<10%）
                    if lane_result and lane_result.get("findings"):
                        result["wb_lane_findings"] = lane_result["findings"]
                except ImportError:
                    pass  # blot_gel_lane_audit 不可用则跳过
        finally:
            try:
                shutil.rmtree(lane_tmp)
            except Exception:
                pass
    
    # ----- SIFT 特征点匹配（旋转/翻转不变）-----
    if HAS_SIFT and result["image_count"] >= 2:
        try:
            # 需要先提取图片到一个目录
            tmp_dir = tempfile.mkdtemp(prefix="sift_")
            try:
                figures_dir = os.path.join(tmp_dir, "figures")
                extracted = extract_figures(pdf_path, figures_dir, min_size=100)
                if len(extracted) >= 2:
                    sift_findings = []
                    for i in range(min(len(extracted), 10)):
                        for j in range(i + 1, min(len(extracted), 10)):
                            try:
                                result_pair = analyze_pair(extracted[i], extracted[j])
                                if result_pair and result_pair.get("match_count", 0) >= 60:  # v8.8.1: 从20升至60，基于阴性对照基线
                                    sift_findings.append(result_pair)
                            except Exception:
                                continue
                    result["sift_findings"] = sift_findings[:10]
            finally:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
        except Exception as e:
            # SIFT 整体失败（如 cv2 不可用、图片全部无法处理）— 跳过不影响其他检测器
            pass

    return result


# ==================================================================
# Step ⑦: 第4+5层全量 — 跨论文+外部验证
# ==================================================================
def step7_cross_paper_external(pdf_paths: list) -> dict:
    """跨论文归因 + 外部验证"""
    result = {
        "paper_count": len(pdf_paths),
        "cross_paper_findings": [],
        "external_findings": [],
        "error": None
    }
    
    if len(pdf_paths) < 2:
        result["note"] = "仅1篇论文，跨论文分析不可用"
        return result
    
    # 第4层：合作署名归因
    if HAS_AUTHOR:
        try:
            detector = AuthorAnomalyDetector()
            # 这里需要从所有论文中提取作者信息进行比对
            # 完整实现需要逐篇提取后运行 detector
            result["note"] = "合作署名归因需要完整实现"
        except Exception as e:
            result["error"] = str(e)
    
    # 第5层：外部验证 — 临床试验注册 + 基金号
    # 这部分需要联网查询 ChiCTR/clinicaltrials.gov/PubMed
    result["external_note"] = "外部验证需要联网查询 ChiCTR/clinicaltrials.gov/PubMed/基金号"
    
    return result


# ==================================================================
# Step ⑧: 独立报告
# ==================================================================
def step8_generate_report(paper_results: list, output_format: str = "json") -> str:
    """生成最终审查报告"""
    
    if output_format == "markdown":
        return _generate_markdown_report(paper_results)
    else:
        return json.dumps(_generate_json_report(paper_results), ensure_ascii=False, indent=2)


def _generate_json_report(paper_results: list) -> dict:
    """生成 JSON 格式报告"""
    
    total_papers = len(paper_results)
    
    # 统计各等级
    grade_counts = defaultdict(int)
    total_findings = 0
    
    for pr in paper_results:
        grade = pr.get("overall_grade", "UNRATED")
        grade_counts[grade] += 1
        total_findings += pr.get("findings_count", 0)
    
    return {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "framework_version": "Heaven's Net v1.2",
            "tool": "heavens-net",
        },
        "summary": {
            "total_papers": total_papers,
            "total_findings": total_findings,
            "grade_distribution": dict(grade_counts),
        },
        "papers": paper_results
    }


def _generate_markdown_report(paper_results: list) -> str:
    """生成 Markdown 格式报告（含图片取证详情）"""
    lines = []
    lines.append("# Heaven's Net 审查报告")
    lines.append(f"**生成时间:** {datetime.now().isoformat()}")
    lines.append(f"**系统版本:** Heaven's Net v1.2")
    lines.append(f"**审查论文数:** {len(paper_results)}")
    lines.append("")
    
    for i, pr in enumerate(paper_results, 1):
        identity = pr.get("step1_identity", {})
        data = pr.get("step2_data_layer", {})
        methods = pr.get("step3_methods", {})
        s6 = pr.get("step6_layer3", {})
        
        lines.append(f"## 论文 {i}: {pr.get('filename', 'unknown')}")
        lines.append("")
        
        # --- 身份 ---
        lines.append("### 基本信息")
        lines.append(f"- **语言:** {identity.get('language', '?')}")
        lines.append(f"- **作者:** {', '.join(identity.get('authors', [])[:5]) or '未提取'}")
        lines.append(f"- **第一作者:** {identity.get('first_author', 'N/A')}")
        lines.append(f"- **通讯作者:** {', '.join(identity.get('corresponding', [])) or '未提取'}")
        lines.append(f"- **单位:** {identity.get('institution', 'N/A')}")
        lines.append(f"- **期刊:** {identity.get('journal', 'N/A')}")
        lines.append(f"- **DOI:** {identity.get('doi', 'N/A')}")
        lines.append(f"- **基金号:** {', '.join(identity.get('funding', [])) or '无'}")
        lines.append("")
        
        # --- 数据层 ---
        lines.append("### 数据层判断")
        lines.append(f"- **数据类型:** {data.get('data_type', 'unknown')}")
        lines.append(f"- **WB图:** {'是' if data.get('has_wb_images') else '否'}")
        lines.append(f"- **临床试验:** {'是' if data.get('is_clinical_trial') else '否'}")
        lines.append(f"- **武器覆盖率:** {data.get('applicable_layers', {})}")
        lines.append("")
        
        # --- 方法 ---
        lines.append("### 方法段")
        if methods.get("p_threshold"):
            lines.append(f"- **P值阈值:** P<{methods['p_threshold']}")
        lines.append(f"- **统计声明:** {', '.join(methods.get('stat_declarations', []))}")
        lines.append(f"- **n值:** {methods.get('n_values', [])}")
        if methods.get("method_violations"):
            lines.append(f"- ⚠️ **方法违反:** {', '.join(methods['method_violations'])}")
        
        # 🆕 v1.1: 三层次统计方法审计
        stat_audit = methods.get("stat_audit", {})
        if stat_audit.get("executed"):
            lines.append("")
            lines.append("#### 三层次统计审计")
            lines.append(f"- **推断数据类型:** {stat_audit.get('inferred_data_type', '?')}")
            lines.append(f"- **提取声明数:** {len(stat_audit.get('declarations', []))}")
            for d in stat_audit.get("declarations", [])[:8]:
                lines.append(f"  - L{d['line']:04d}: {d['method']}")
            
            audit_findings = stat_audit.get("findings", [])
            if audit_findings:
                lines.append(f"- **发现:** HIGH={stat_audit.get('high_count',0)}, MEDIUM={stat_audit.get('med_count',0)}")
                for f in audit_findings[:10]:
                    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(f.get("severity",""), "•")
                    lines.append(f"  - {icon} [第{f['layer']}层:{f.get('category','')}] {f['description'][:120]}")
        
        lines.append("")
        
        # --- 图片取证 ---
        lines.append("### 图片取证（Step ⑥ 第3层）")
        img_count = s6.get("image_count", 0)
        extracted = s6.get("extracted_images", [])
        lines.append(f"- **PDF嵌入图总数:** {img_count}")
        lines.append(f"- **提取分析图数:** {len(extracted)}")
        lines.append("")
        
        # 🆕 v8.4: aHash 初筛
        ahash = s6.get("ahash_findings", [])
        if ahash:
            lines.append("#### aHash 初筛（图片重复候选）")
            high_ah = sum(1 for f in ahash if f.get("severity") == "HIGH")
            med_ah = sum(1 for f in ahash if f.get("severity") == "MEDIUM")
            lines.append(f"- 🔴 HIGH: {high_ah} 对")
            lines.append(f"- 🟡 MEDIUM: {med_ah} 对")
            lines.append("> ⚠️ aHash 是初筛，所有候选需经 ELA/SIFT 二次验证。")
            lines.append("")
        
        # ELA/块重复/背景突变
        ela_count = len(s6.get("ela_findings", []))
        block_count = len(s6.get("block_dup_findings", []))
        bg_count = len(s6.get("background_mutation_findings", []))
        if ela_count or block_count or bg_count:
            lines.append("#### 图片取证（ELA + 块重复 + 背景突变）")
            if ela_count:
                lines.append(f"- **ELA异常:** {ela_count} 处")
                for f in s6.get("ela_findings", [])[:5]:
                    lines.append(f"  - ELA={f.get('ela_score', '?')}, 文件: {Path(f.get('file','?')).name}")
            if block_count:
                lines.append(f"- **块重复:** {block_count} 处")
            if bg_count:
                lines.append(f"- **背景突变:** {bg_count} 处")
            lines.append("")
        else:
            lines.append("#### ELA/块重复/背景突变: 未发现异常")
            lines.append("")
        
        # pHash 跨图比对
        cross_img = s6.get("cross_image_findings", [])
        if cross_img:
            lines.append("#### 同论文内跨图比对（pHash）")
            high_pairs = [c for c in cross_img if c.get("level") == "HIGH"]
            mod_pairs = [c for c in cross_img if c.get("level") == "MODERATE"]
            if high_pairs:
                lines.append(f"- 🔴 **高度相似 (HIGH):** {len(high_pairs)} 对")
                for c in high_pairs[:5]:
                    lines.append(f"  - {Path(c['fig1']).name} ↔ {Path(c['fig2']).name}: sim={c['similarity']}")
            if mod_pairs:
                lines.append(f"- 🟡 **中度相似 (MODERATE):** {len(mod_pairs)} 对")
            lines.append("> ⚠️ pHash假阳性陷阱: PDF整页提取的图片以白底为主，自然相似度偏高。HIGH级需SSIM二次验证。")
            lines.append("")
        else:
            lines.append("#### 跨图比对: 未发现显著相似")
            lines.append("")
        
        # WB
        wb = s6.get("wb_findings", [])
        if wb:
            lines.append("#### Western Blot 条带分析")
            lines.append(f"- **WB发现:** {len(wb)} 条")
            for f in wb[:5]:
                lines.append(f"  - {json.dumps(f, ensure_ascii=False)[:150]}")
            lines.append("")

        # WB 泳道检测
        wb_lane = s6.get("wb_lane_findings", [])
        if wb_lane:
            lines.append("#### WB 泳道拼接/重复检测")
            boundary_f = [f for f in wb_lane if "boundary" in f.get("method", "")]
            lane_f = [f for f in wb_lane if "repeated_lane" in f.get("method", "")]
            if boundary_f:
                lines.append(f"- **分界线（拼接痕迹）:** {len(boundary_f)} 处")
            if lane_f:
                lines.append(f"- **重复泳道:** {len(lane_f)} 对")
            lines.append("")
        
        # SIFT
        sift = s6.get("sift_findings", [])
        if sift:
            lines.append("#### SIFT 特征点匹配")
            lines.append(f"- **SIFT匹配对:** {len(sift)} 对")
            lines.append("> ⚠️ **SIFT整页假阳性警告:** PDF整页提取的图片包含大量共同的PDF渲染元素（线条、文字、边框），")
            lines.append("> SIFT匹配点数高不等同于图片内容重复。需要bbox级图片裁剪才能排除假阳性。")
            lines.append("> 当前SIFT信号仅为筛查参考，不能独立定罪。")
            lines.append("")
        
        # 文本异常
        text_f = s6.get("text_findings", [])
        if text_f:
            lines.append("#### 文本异常")
            for f in text_f[:5]:
                lines.append(f"- {f}")
            lines.append("")
        
        # Step ④⑤
        s4 = pr.get("step4_layer1", {})
        if s4.get("mean_sd_pairs_found"):
            lines.append("### 统计反算（Step ④）")
            lines.append(f"- **Mean±SD对:** {s4['mean_sd_pairs_found']} 对")
            lines.append("")
        
        # 总体
        lines.append("### 总体")
        lines.append(f"- **发现信号数:** {pr.get('findings_count', 0)}")
        lines.append(f"- **等级:** {pr.get('overall_grade', 'PENDING')}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return '\n'.join(lines)


# ==================================================================
# 主函数：逐篇八步审查流程
# ==================================================================
def review_paper(pdf_path: str, all_paths: list = None) -> dict:
    """对单篇论文执行八步审查"""
    
    filename = Path(pdf_path).name
    print(f"  [{filename}] 开始审查...")
    
    result = {
        "filename": filename,
        "path": str(pdf_path),
        "steps_completed": [],
        "findings_count": 0,
        "overall_grade": "PENDING",
    }
    
    # Step ① 身份确认
    print(f"    Step ① 身份确认...")
    result["step1_identity"] = step1_identity(pdf_path)
    result["steps_completed"].append("①")
    
    # Step ② 数据层判断
    print(f"    Step ② 数据层判断... → {result['step1_identity'].get('data_type', '?')}")
    # 注意: step2_data_layer 是独立的函数，这里调用需要在上面定义
    # 字段名: step2_data_layer
    data_result = step2_data_layer(pdf_path)
    result["step2_data_layer"] = data_result
    result["steps_completed"].append("②")
    
    # Step ③ 方法
    print(f"    Step ③ 逐段读方法...")
    result["step3_methods"] = step3_methods(pdf_path)
    result["steps_completed"].append("③")
    
    # Step ③½ 跨表数据联动（同篇论文不同表格的同一组数据是否一致）
    if HAS_CROSS_TABLE:
        print(f"    Step ③½ 跨表数据联动...")
        try:
            cross_table_checker = CrossTableConsistencyChecker()
            cross_table_checker.extract_from_pdf(pdf_path)
            cross_table_checker.check_consistency()
            result["step3a_cross_table"] = cross_table_checker.to_dict()
            result["steps_completed"].append("③½")
        except Exception as e:
            result["step3a_cross_table"] = {"error": str(e)}
    else:
        result["step3a_cross_table"] = {"error": "cross_table_consistency not available"}
    
    # Step ④ 第1层
    print(f"    Step ④ 第1层统计反算...")
    result["step4_layer1"] = step4_layer1(pdf_path)
    result["steps_completed"].append("④")
    
    # Step ④½ 时间序列虚假模式检测
    if HAS_TIME_SERIES:
        print(f"    Step ④½ 时间序列检测...")
        try:
            ts_detector = TimeSeriesFraudDetector()
            ts_detector.extract_from_pdf(pdf_path)
            ts_detector.detect_all()
            result["step4a_time_series"] = ts_detector.to_dict()
            result["steps_completed"].append("④½")
        except Exception as e:
            result["step4a_time_series"] = {"error": str(e)}
    else:
        result["step4a_time_series"] = {"error": "time_series_fraud not available"}
    
    # Step ⑤ 第2层
    print(f"    Step ⑤ 第2层算术模式...")
    result["step5_layer2"] = step5_layer2(pdf_path)
    result["steps_completed"].append("⑤")
    
    # Step ⑥ 第3层
    print(f"    Step ⑥ 第3层图片+文本...")
    result["step6_layer3"] = step6_layer3(pdf_path)
    result["steps_completed"].append("⑥")
    
    # Step ⑦ 跨论文+外部验证（需要全部论文一起跑，暂跳过单篇）
    if all_paths and len(all_paths) > 1:
        result["step7_cross_paper"] = {"note": "跨论文分析在全部论文审查后统一执行"}
    else:
        result["step7_cross_paper"] = {"note": "仅1篇论文，跨论文分析不适用"}
    result["steps_completed"].append("⑦")
    
    # 汇总 findings（计数各层的发现）
    findings_count = 0
    for key in result:
        val = result[key]
        if isinstance(val, dict):
            # 计数 findings 字段
            fd = val.get("findings", [])
            if isinstance(fd, list):
                findings_count += len(fd)
            # 计数各图片取证字段
            for img_key in ["ahash_findings", "ela_findings", "block_dup_findings", "background_mutation_findings",
                           "wb_lane_findings", "wb_findings", "cross_image_findings", "sift_findings", "text_findings"]:
                img_fd = val.get(img_key, [])
                if isinstance(img_fd, list):
                    findings_count += len(img_fd)
            # 🆕 跨表联动
            ct_findings = val.get("findings", val.get("total_findings", 0))
            # 🆕 时间序列
            ts_findings = val.get("total_findings", 0)
    # 手动加上跨表和时间序列的 findings（key 为 step3a_cross_table 和 step4a_time_series）
    ct = result.get("step3a_cross_table", {})
    if isinstance(ct, dict):
        findings_count += ct.get("total_findings", 0)
    ts = result.get("step4a_time_series", {})
    if isinstance(ts, dict):
        findings_count += ts.get("total_findings", 0)
    result["findings_count"] = findings_count
    
    print(f"    → 发现 {findings_count} 个信号")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Heaven's Net — 学术不端检测系统 v1.2 | 天网恢恢，疏而不漏",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  heavens-net ~/论文目录/
  heavens-net paper.pdf --output report.json
  heavens-net ~/论文目录/ --format markdown --output report.md
  heavens-net ~/论文目录/ --cross-file
        """
    )
    parser.add_argument("input", help="PDF文件路径或包含多个PDF的目录路径")
    parser.add_argument("--output", "-o", help="输出文件路径（默认: review_report_<时间戳>.json）")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="json",
                       help="输出格式 (默认: json)")
    parser.add_argument("--layers", "-l", help="仅执行指定层 (如: 1,2,5)")
    parser.add_argument("--cross-file", "-x", action="store_true", 
                       help="启用跨文件检测模式")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--version", action="version", version="Heaven's Net v1.2")
    
    args = parser.parse_args()
    
    input_path = Path(args.input).expanduser().resolve()
    
    if not input_path.exists():
        print(f"❌ 路径不存在: {input_path}")
        sys.exit(1)
    
    # 收集所有 PDF 文件
    pdf_files = []
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        pdf_files = [str(input_path)]
    elif input_path.is_dir():
        pdf_files = sorted([str(p) for p in input_path.glob("*.pdf")])
        # 也搜索子目录
        pdf_files.extend(sorted([str(p) for p in input_path.rglob("*.pdf")]))
        pdf_files = list(dict.fromkeys(pdf_files))  # 去重保持顺序
    else:
        print(f"❌ 请提供 PDF 文件或包含 PDF 的目录")
        sys.exit(1)
    
    if not pdf_files:
        print(f"❌ 未找到 PDF 文件")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Heaven's Net v1.2 — 天网恢恢，疏而不漏")
    print(f"论文数: {len(pdf_files)}")
    print(f"输入路径: {input_path}")
    print(f"{'='*60}\n")
    
    # 逐篇审查
    paper_results = []
    start_time = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}]")
        result = review_paper(pdf_path, pdf_files)
        paper_results.append(result)
    
    # 如果多篇，执行跨论文分析
    if len(pdf_files) > 1:
        print(f"\n执行跨论文分析（Step ⑦ 第4+5层）...")
        cross_result = step7_cross_paper_external(pdf_files)
        
        # 跨论文图片比对（bulk_cross_paper_analysis）
        if HAS_CROSS_PAPER_IMG and args.cross_file:
            print(f"  跨论文图片比对（pHash + SSIM）...")
            try:
                # 提取所有论文的图片
                all_figures = {}
                import tempfile
                tmp_dir = tempfile.mkdtemp(prefix="cross_paper_")
                for pdf_path in pdf_files:
                    paper_name = Path(pdf_path).stem
                    fig_dir = os.path.join(tmp_dir, paper_name)
                    try:
                        figs = extract_figures(pdf_path, fig_dir, min_size=100)
                        if figs:
                            all_figures[paper_name] = figs
                    except Exception:
                        pass
                
                if len(all_figures) >= 2:
                    cross_img_findings = cross_paper_similarity(all_figures)
                    if cross_img_findings:
                        cross_result["cross_paper_image_similarity"] = cross_img_findings[:20]
                
                # 清理
                import shutil
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            except Exception as e:
                cross_result["cross_paper_image_error"] = str(e)
        
        # 附加到最后一篇论文的结果中
        if paper_results:
            paper_results[-1]["step7_cross_paper_full"] = cross_result
    
    elapsed = time.time() - start_time
    
    # 生成报告
    report = step8_generate_report(paper_results, args.format)
    
    # 输出文件
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if args.format == "markdown" else "json"
        output_path = f"review_report_{timestamp}.{ext}"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 摘要
    print(f"\n{'='*60}")
    print(f"审查完成")
    print(f"  论文数: {len(pdf_files)}")
    print(f"  耗时: {elapsed:.1f}秒")
    print(f"  报告: {output_path}")
    print(f"{'='*60}")
    
    # 简要统计
    if args.format == "json":
        report_data = json.loads(report)
        grade_dist = report_data.get("summary", {}).get("grade_distribution", {})
        if grade_dist:
            print(f"\n等级分布:")
            for grade, count in sorted(grade_dist.items()):
                print(f"  {grade}: {count}篇")


if __name__ == '__main__':
    main()
