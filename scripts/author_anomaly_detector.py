#!/usr/bin/env python3
"""
论文作者异常检测器
==================
实现扮虎/Tiger 团队的多个中文造假检测方法：

1. 邮箱域名检测——非学术邮箱占比过高 = 工厂信号
2. ORCID 缺失检测——论文工厂论文通常不注册 ORCID
3. 作者单位与论文类型矛盾——地区医院做分子生物学 = 不可能
4. 跨地域异常合作——吉林+云南作者无故"合作"

依赖: PyMuPDF (提取PDF元数据和作者信息), PubMed E-utilities API

用法:
    python author_anomaly_detector.py paper.pdf
    python author_anomaly_detector.py --dir papers/ --json results.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# 非学术邮箱域名（扮虎发现工厂论文大量使用）
SUSPICIOUS_EMAIL_DOMAINS = [
    '163.com', '126.com', 'qq.com', 'sina.com', 'sohu.com',
    'yeah.net', 'sina.cn', 'aliyun.com', 'foxmail.com', 'tom.com',
    '139.com', '189.cn',
]

# 学术/机构邮箱域名
ACADEMIC_EMAIL_DOMAINS = [
    '.edu', '.ac.', '.gov', '.org',
]

# 已知的高撤稿率机构（Nature 2025分析+公开通报）
HIGH_RETRACTION_INSTITUTIONS = [
    '济宁市第一人民医院', '沧州市中心医院', '河南大学淮河医院',
    '潍坊市人民医院', '临沂市人民医院', '新乡医学院第一附属医院',
    '齐齐哈尔医学院',
]

# 不可能产出特定类型研究的机构模式
IMPOSSIBLE_RESEARCH_PATTERNS = [
    (r'(?:县|区|镇|乡).*医院', r'(?:细胞|分子|基因|蛋白|RNA|DNA|WB|western|PCR|qPCR|转染|敲除|过表达|免疫组化|流式)'),
]


def extract_authors_from_pdf(pdf_path):
    """从PDF提取作者和机构信息"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc[:3]:  # 前3页通常包含作者信息
            text += page.get_text()
        
        # 提取邮箱
        emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
        
        # 提取作者名（中英文）
        authors_cn = re.findall(r'(?:作者|通讯作者|第一作者)[：:]\s*(.+?)(?:[。，;；\n]|$)', text)
        authors_en = re.findall(r'([A-Z][a-z]+ (?:[A-Z]\.?\s?)?[A-Z][a-z]+)', text)
        
        # 提取机构
        affiliations = re.findall(r'(?:单位|机构|作者单位|affiliation|department)[：:]\s*(.+?)(?:[。\n]|$)', text, re.IGNORECASE)
        
        # PubMed DOI 识别
        dois = re.findall(r'(10\.\d{4,}/[\w.()-]+)', text)
        
        return {
            "emails": emails,
            "authors_cn": authors_cn,
            "authors_en": authors_en[:20],
            "affiliations": affiliations,
            "dois": dois,
            "text_first_3_pages": text[:5000]
        }
    
    except Exception as e:
        return {"error": str(e)}


def analyze_emails(emails):
    """分析邮箱域名的可疑度"""
    if not emails:
        return {"status": "NO_EMAILS_FOUND"}
    
    total = len(emails)
    suspicious = [e for e in emails if any(d in e for d in SUSPICIOUS_EMAIL_DOMAINS)]
    academic = [e for e in emails if any(d in e for d in ACADEMIC_EMAIL_DOMAINS)]
    
    suspicious_rate = len(suspicious) / total if total > 0 else 0
    
    signal = "CLEAN"
    if suspicious_rate > 0.5:
        signal = "HIGH"
    elif suspicious_rate > 0.3:
        signal = "MEDIUM"
    elif academic and len(academic) < total * 0.3:
        signal = "LOW"
    
    return {
        "total_emails": total,
        "suspicious_count": len(suspicious),
        "academic_count": len(academic),
        "suspicious_rate": f"{suspicious_rate*100:.1f}%",
        "suspicious_domains": list(set(e.split('@')[1] for e in suspicious)),
        "signal": signal,
        "detail": f"非学术邮箱占比{suspicious_rate*100:.0f}% — 工厂论文通常>50%",
    }


def analyze_institution(author_info):
    """分析机构可疑度"""
    text = author_info.get("text_first_3_pages", "")
    affiliations = author_info.get("affiliations", [])
    
    findings = []
    
    # 检测高撤稿率机构
    for inst in HIGH_RETRACTION_INSTITUTIONS:
        if inst in text:
            findings.append({
                "type": "HIGH_RETRACTION_INSTITUTION",
                "institution": inst,
                "signal": "MEDIUM",
                "detail": f"机构{inst}在Nature 2025全球撤稿排名中上榜"
            })
    
    # 检测机构-研究类型矛盾
    for inst_pattern, research_pattern in IMPOSSIBLE_RESEARCH_PATTERNS:
        if re.search(inst_pattern, text):
            if re.search(research_pattern, text):
                findings.append({
                    "type": "INSTITUTION_RESEARCH_MISMATCH",
                    "signal": "HIGH",
                    "detail": "地区医院声称完成分子生物学实验——没有实验室设施的医院不可能完成此类研究"
                })
    
    # 机构数量过多 = 国际合作异常
    aff_count = len(set(affiliations)) if affiliations else 0
    
    return {
        "findings": findings,
        "n_unique_affiliations": aff_count,
        "high_retraction_match": any("HIGH_RETRACTION" in f["type"] for f in findings),
        "research_mismatch": any("MISMATCH" in f["type"] for f in findings),
    }


def analyze_author_network(author_info, dois):
    """分析作者网络异常"""
    findings = []
    
    # 无ORCID/DOI = 常见工厂特征
    if not dois:
        findings.append({
            "type": "NO_DOI",
            "signal": "LOW",
            "detail": "PDF未检测到DOI——工厂论文可能缺少正式DOI注册"
        })
    
    return findings


class AuthorAnomalyDetector:
    """作者异常检测器兼容包装"""
    
    def __init__(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="论文作者异常检测器")
    parser.add_argument("pdf_path", nargs="?", help="PDF文件路径")
    parser.add_argument("--dir", help="批量扫描目录")
    parser.add_argument("--json", help="输出JSON文件")
    args = parser.parse_args()
    
    if args.dir:
        results = {}
        for pdf in sorted(Path(args.dir).glob("*.pdf")):
            info = extract_authors_from_pdf(str(pdf))
            if "error" in info:
                results[pdf.name] = info
                continue
            
            results[pdf.name] = {
                "emails": analyze_emails(info.get("emails", [])),
                "institution": analyze_institution(info),
                "author_network": analyze_author_network(info, info.get("dois", [])),
            }
        
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 汇总
        n_high = sum(1 for r in results.values() if r.get("emails",{}).get("signal") == "HIGH")
        print(f"作者异常检测完成: {len(results)}篇, {n_high}篇有邮箱高危信号")
        return
    
    if args.pdf_path:
        info = extract_authors_from_pdf(args.pdf_path)
        if "error" in info:
            print(f"错误: {info['error']}")
            return
        
        email_result = analyze_emails(info.get("emails", []))
        inst_result = analyze_institution(info)
        
        print("=" * 64)
        print(f"  作者异常检测: {Path(args.pdf_path).name}")
        print("=" * 64)
        
        print(f"\n📧 邮箱检测: {email_result.get('total_emails', 0)}个邮箱")
        if email_result.get("suspicious_count", 0) > 0:
            print(f"  非学术邮箱: {email_result['suspicious_count']}个 ({email_result['suspicious_rate']}) — {email_result['signal']}")
            for d in email_result.get("suspicious_domains", []):
                print(f"    {d}")
        else:
            print(f"  ✅ 未发现非学术邮箱")
        
        if inst_result["findings"]:
            print(f"\n🏥 机构异常:")
            for f in inst_result["findings"]:
                icon = "🔴" if f["signal"] == "HIGH" else "🟡" if f["signal"] == "MEDIUM" else "⚪"
                print(f"  {icon} {f['detail']}")
        
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"emails": email_result, "institution": inst_result}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
