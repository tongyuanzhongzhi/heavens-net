#!/usr/bin/env python3
"""
中文论文统计格式解析器
=======================
检测中文论文特有的造假信号，弥补英文系统中
全部基于英文设计的工具在中文面前的盲区。

检测项：
1. 小数点位数的异常集中（检测器标志性方法）
2. P值格式违规的中文变体
3. 精确P值完全缺失检测
4. 相同数值在"独立"测量中的重复出现
5. 统计描述格式解析

用法:
    python chinese_stat_parser.py paper.pdf
    python chinese_stat_parser.py --dir papers/ --json results.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """从PDF提取中文文本"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"PDF_EXTRACT_ERROR: {e}"


def detect_decimal_concentration(text):
    """
    检测小数点位数的异常集中
    
    检测器打假的经典案例：
    "13个数据11个小数点后两位相同，其中6个完全相同"
    
    这是人工编造数据时无法避免的心理惯性——造假者会下意识重复使用
    相同的尾数，而不是随机分布。
    """
    # 提取所有带小数的数字
    decimal_pattern = re.compile(r'\b(\d+\.(\d+))\b')
    matches = decimal_pattern.findall(text)
    
    if not matches:
        return {"status": "INSUFFICIENT_DATA", "n_decimals": 0}
    
    # 按小数位数分组
    by_digits = {}
    for full_match, decimal_part in matches:
        n_digits = len(decimal_part)
        if n_digits not in by_digits:
            by_digits[n_digits] = []
        by_digits[n_digits].append(decimal_part)
    
    findings = []
    for n_digits, decimals in by_digits.items():
        if len(decimals) < 10:
            continue
        
        # 统计数字频率
        total = len(decimals)
        counter = Counter(decimals)
        most_common = counter.most_common(5)
        
        # 如果某种尾数出现频率过高
        max_pct = most_common[0][1] / total if most_common else 0
        
        signal = "CLEAN"
        if max_pct > 0.5:
            signal = "HIGH"
        elif max_pct > 0.3:
            signal = "MEDIUM"
        
        if signal != "CLEAN" or n_digits <= 2:
            findings.append({
                "decimal_digits": n_digits,
                "total_decimals": total,
                "most_common_values": [(v, c, f"{c/total*100:.1f}%") for v, c in most_common[:3]],
                "max_concentration_pct": f"{max_pct*100:.1f}%",
                "signal": signal,
                "note": f"{n_digits}位小数, 最常见='{most_common[0][0]}' 出现{most_common[0][1]}次({max_pct*100:.1f}%)"
            })
    
    return {
        "n_decimals": sum(len(v) for v in by_digits.values()),
        "decimal_types": len(by_digits),
        "findings": findings,
        "status": "ANALYZED"
    }


def detect_pvalue_violations(text):
    """
    检测中文P值格式违规
    
    中文论文常见的P值问题：
    1. 全文只有P<0.05无精确值
    2. P=0.000（SPSS默认输出）
    3. P<0.000（无效表示）
    4. "差异有统计学意义(P<0.05)"格式——statcheck无法解析
    """
    # P值格式检测（已在statcheck中覆盖，这里做中文特化）
    p_exact_zero = re.findall(r'P\s*=\s*0\.000(?!\d)', text)
    p_lt_zero = re.findall(r'P\s*<\s*0\.000', text)
    
    # 统计所有P值出现
    p_all_threshold = len(re.findall(r'P\s*<\s*0\.\d+', text))
    p_all_exact = len(re.findall(r'P\s*=\s*0\.\d+', text))
    
    # 检测是否有任何精确P值（>3位小数）
    p_precise = re.findall(r'P\s*=\s*0\.\d{3,}', text)
    
    # 中文特有格式
    cn_significant = len(re.findall(r'差异有统计学意义', text))
    cn_p_value = len(re.findall(r'P[值<]', text))
    
    findings = []
    
    if p_exact_zero:
        findings.append({
            "type": "P=0.000",
            "count": len(p_exact_zero),
            "severity": "L2",
            "detail": f"{len(p_exact_zero)}处P=0.000（SPSS默认输出未修改）",
            "examples": p_exact_zero[:3]
        })
    
    if p_lt_zero:
        findings.append({
            "type": "P<0.000",
            "count": len(p_lt_zero),
            "severity": "L2",
            "detail": f"{len(p_lt_zero)}处P<0.000（无效表示法）",
            "examples": p_lt_zero[:3]
        })
    
    # 全文只有阈值无精确值
    if p_all_threshold > 5 and p_all_exact == 0:
        findings.append({
            "type": "NO_PRECISE_P_VALUES",
            "count": p_all_threshold,
            "severity": "L1",
            "detail": f"全文{p_all_threshold}处P值全部使用阈值格式（如P<0.05），无任何精确P值"
        })
    
    return {
        "p_exact_zero_count": len(p_exact_zero),
        "p_lt_zero_count": len(p_lt_zero),
        "p_threshold_only": p_all_threshold,
        "p_exact": p_all_exact,
        "p_precise_count": len(p_precise),
        "cn_significant_count": cn_significant,
        "findings": findings,
        "statcheck_compatible": p_all_exact > 0,
    }


def detect_duplicate_values(text):
    """
    检测相同数值在不同"独立"测量中的重复出现
    
    检测器案例："13个数据8个相同，剩下5个高度相似"
    """
    # 提取所有数值
    numbers = re.findall(r'\b(\d+\.?\d*)\b', text)
    
    if len(numbers) < 10:
        return {"status": "INSUFFICIENT_DATA"}
    
    # 只分析有意义长度的数字（≥3位）
    significant = [n for n in numbers if len(n.replace('.','')) >= 3]
    
    if len(significant) < 10:
        return {"status": "INSUFFICIENT_DATA"}
    
    counter = Counter(significant)
    duplicates = [(v, c) for v, c in counter.most_common(20) if c >= 2]
    
    # 计算重复率
    dup_rate = sum(c for _, c in duplicates) / len(significant) if significant else 0
    
    signal = "CLEAN"
    if dup_rate > 0.3:
        signal = "HIGH"
    elif dup_rate > 0.15:
        signal = "MEDIUM"
    
    return {
        "n_significant_numbers": len(significant),
        "n_unique": len(counter),
        "n_duplicated": len(significant) - len(counter),
        "duplication_rate": f"{dup_rate*100:.1f}%",
        "top_duplicates": [(v, c, f"{c/len(significant)*100:.1f}%") for v, c in duplicates[:5]],
        "signal": signal,
        "status": "ANALYZED"
    }


def main():
    parser = argparse.ArgumentParser(description="中文论文统计格式解析器")
    parser.add_argument("pdf_path", nargs="?", help="PDF文件路径")
    parser.add_argument("--dir", help="批量扫描目录")
    parser.add_argument("--json", help="输出JSON文件")
    parser.add_argument("--text", help="直接分析文本字符串")
    args = parser.parse_args()
    
    results = {}
    
    if args.text:
        results = analyze_text(args.text)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    
    if args.dir:
        pdf_dir = Path(args.dir)
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        all_results = {}
        
        for pdf in pdfs:
            text = extract_text_from_pdf(str(pdf))
            if text.startswith("PDF_EXTRACT_ERROR"):
                all_results[pdf.name] = {"error": text}
                continue
            all_results[pdf.name] = analyze_text(text)
        
        if args.json:
            with open(args.json, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # Summary
        n_papers = len(all_results)
        n_issues = sum(1 for r in all_results.values() if any(
            f.get("signal") in ("HIGH", "MEDIUM") 
            for f in r.get("decimal_concentration", {}).get("findings", [])
        ))
        print(f"中文统计解析完成: {n_papers}篇, {n_issues}篇有显著信号")
        return
    
    if args.pdf_path:
        text = extract_text_from_pdf(args.pdf_path)
        if text.startswith("PDF_EXTRACT_ERROR"):
            print(text)
            return
        results = analyze_text(text)
        
        # Output
        print("=" * 64)
        print(f"  中文统计格式解析: {Path(args.pdf_path).name}")
        print("=" * 64)
        
        dc = results["decimal_concentration"]
        print(f"\n📊 小数点位集中度: {dc['n_decimals']}个小数, {dc['decimal_types']}种位数")
        for f in dc.get("findings", []):
            signal_icon = "🔴" if f["signal"] == "HIGH" else "🟡" if f["signal"] == "MEDIUM" else "🟢"
            print(f"  {signal_icon} {f['note']}")
        
        pv = results["pvalue_violations"]
        print(f"\n📏 P值格式: P=0.000={pv['p_exact_zero_count']}, P<0.000={pv['p_lt_zero_count']}")
        print(f"  精确P值: {pv['p_precise_count']}个 | Statcheck兼容: {pv['statcheck_compatible']}")
        for f in pv["findings"]:
            print(f"  ⚠️ {f['detail']}")
        
        dv = results["duplicate_values"]
        if dv.get("status") == "ANALYZED":
            print(f"\n🔁 数值重复: {dv['duplication_rate']}重复率, {dv['n_duplicated']}个重复值")
            if dv["signal"] != "CLEAN":
                print(f"  ⚠️ {dv['signal']}: Top重复值: {dv['top_duplicates'][:3]}")
        
        if args.json:
            with open(args.json, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)


def analyze_text(text):
    """对提取的文本运行所有中文检测"""
    return {
        "text_length": len(text),
        "decimal_concentration": detect_decimal_concentration(text),
        "pvalue_violations": detect_pvalue_violations(text),
        "duplicate_values": detect_duplicate_values(text),
    }


if __name__ == "__main__":
    main()
