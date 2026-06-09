#!/usr/bin/env python3
"""
跨论文邮箱关联检测器
====================
实现王景周/扮虎的核心方法：
1. 同一邮箱在不同论文中出现（论文工厂客服的统一联系方式）
2. 邮箱命名模板化规律（"人名+数字"模式 = 批量注册特征）
3. 通讯作者邮箱与其它论文作者重合

这是真正有意义的信号——不是邮箱后缀本身有问题，
而是同一邮箱被不同身份复用才说明问题。

用法:
    python cross_paper_email_analyzer.py --dir papers/
    python cross_paper_email_analyzer.py --db emails_db.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path


def extract_email_db_from_pdfs(papers_dir):
    """从论文目录批量提取邮箱数据库"""
    import fitz
    
    paper_emails = {}
    all_pdfs = []
    for root, dirs, files in os.walk(papers_dir):
        for f in files:
            if f.endswith('.pdf') and not f.startswith('.'):
                all_pdfs.append(os.path.join(root, f))
    
    for pdf_path in all_pdfs:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc[:5]:
            full_text += page.get_text()
        
        emails = list(set(re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', full_text)))
        paper_name = os.path.basename(pdf_path).replace('.pdf','')
        paper_emails[paper_name] = {
            "file": pdf_path,
            "emails": emails,
            "n_emails": len(emails),
        }
    
    return paper_emails


def build_cross_paper_map(paper_emails):
    """构建邮箱→论文的跨论文映射"""
    email_to_papers = defaultdict(list)
    for paper_name, info in paper_emails.items():
        for email in info["emails"]:
            email_to_papers[email].append(paper_name)
    
    # 只保留出现在多篇论文中的邮箱
    return {e: papers for e, papers in email_to_papers.items() if len(papers) > 1}


def detect_pattern_emails(paper_emails):
    """检测邮箱命名模板化规律"""
    # 模式1: 人名+数字 (zhangsan66, li_si88, wang5)
    pattern_name_number = re.compile(r'^([a-z]{3,}_?[a-z]*)(\d{2,5})@', re.IGNORECASE)
    # 模式2: 纯缩写+数字 (ymh3011, lxj068)
    pattern_abbr_number = re.compile(r'^([a-z]{2,6})(\d{3,6})@', re.IGNORECASE)
    
    findings = []
    
    for paper_name, info in paper_emails.items():
        emails = info["emails"]
        if len(emails) < 2:
            continue
        
        patterned = []
        for e in emails:
            m = pattern_name_number.match(e) or pattern_abbr_number.match(e)
            if m:
                patterned.append(e)
        
        # 同论文内多个邮箱使用相同模板
        if len(patterned) >= 2:
            findings.append({
                "paper": paper_name,
                "type": "PATTERNED_NAMING",
                "detail": f"{len(patterned)}/{len(emails)}邮箱使用'人名/缩写+数字'模板",
                "emails": patterned,
                "signal": "MEDIUM",
                "explanation": "模板化命名暗示批量注册——独立作者不会不约而同选择相同命名模式"
            })
    
    return findings


def detect_cross_paper_signals(cross_paper_emails, paper_emails):
    """分析跨论文同邮箱的真正含义"""
    findings = []
    
    for email, papers in cross_paper_emails.items():
        n_papers = len(papers)
        
        # 核心问题：这个邮箱的主人到底是谁？
        # 如果同一邮箱在A论文是通讯作者，在B论文是普通作者 → 可能是通讯作者在跨论文投稿
        # 如果同一邮箱在A论文是作者张三，在B论文是作者李四 → 论文工厂客服统一联系方式
        
        # 我们目前只能提取到邮箱，不能提取到"这个邮箱属于谁"
        # 所以需要标注：需要人工核实该邮箱的实际归属
        
        findings.append({
            "type": "CROSS_PAPER_SAME_EMAIL",
            "email": email,
            "n_papers": n_papers,
            "papers": papers,
            "signal": "NEEDS_HUMAN_VERIFICATION",
            "detail": f"邮箱{email}出现在{n_papers}篇论文中。需核实：该邮箱在每篇论文中是否属于同一人？",
            "verification_question": f"请检查{email}在各论文中对应的作者名是否一致。"
        })
    
    return findings


def analyze_results(paper_emails, cross_paper_emails, pattern_findings):
    """汇总分析结果"""
    summary = {
        "n_papers": len(paper_emails),
        "n_total_emails": sum(len(v["emails"]) for v in paper_emails.values()),
        "n_cross_paper_emails": len(cross_paper_emails),
        "n_pattern_findings": len(pattern_findings),
    }
    
    # 计算每篇论文的风险信号数
    paper_signals = defaultdict(list)
    
    for email, papers in cross_paper_emails.items():
        for p in papers:
            paper_signals[p].append(f"跨论文邮箱: {email}")
    
    for pf in pattern_findings:
        paper_signals[pf["paper"]].append(f"命名模板: {pf['detail']}")
    
    return summary, dict(paper_signals)


class CrossPaperEmailAnalyzer:
    """跨论文邮箱关联检测器兼容包装"""
    
    def __init__(self):
        pass


def main():
    parser = argparse.ArgumentParser(description="跨论文邮箱关联检测器")
    parser.add_argument("--dir", help="论文PDF目录")
    parser.add_argument("--db", help="已有的邮箱数据库JSON")
    parser.add_argument("--json", help="输出完整JSON")
    args = parser.parse_args()
    
    if args.db:
        with open(args.db) as f:
            data = json.load(f)
            paper_emails = data.get("papers", data)
    elif args.dir:
        paper_emails = extract_email_db_from_pdfs(args.dir)
    else:
        print("请指定 --dir 或 --db")
        sys.exit(1)
    
    cross_paper = build_cross_paper_map(paper_emails)
    patterns = detect_pattern_emails(paper_emails)
    cross_signals = detect_cross_paper_signals(cross_paper, paper_emails)
    summary, paper_signals = analyze_results(paper_emails, cross_paper, patterns)
    
    # 输出
    print("=" * 64)
    print("  跨论文邮箱关联检测")
    print("=" * 64)
    print(f"\n  {summary['n_papers']}篇论文, {summary['n_total_emails']}个邮箱")
    
    # 跨论文同邮箱（需要人工核实）
    if cross_paper:
        print(f"\n{'─'*60}")
        print(f"  ⚡ 跨论文同邮箱: {len(cross_paper)}个邮箱出现在多篇论文中")
        print(f"  ⚠️ 需要人工核实：同一个邮箱在各论文中是否属于同一人？")
        print(f"{'─'*60}")
        
        for email, papers in sorted(cross_paper.items(), key=lambda x: len(x[1]), reverse=True):
            icon = "🔴" if len(papers) >= 3 else "🟡"
            print(f"\n  {icon} {email} — {len(papers)}篇论文:")
            for p in papers:
                # 显示该论文中此邮箱的上下文
                context = ""
                print(f"     └─ {p}{context}")
    
    # 命名模板化
    if patterns:
        print(f"\n{'─'*60}")
        print(f"  📐 邮箱命名模板化: {len(patterns)}篇论文")
        print(f"{'─'*60}")
        
        for pf in patterns:
            print(f"\n  🟡 {pf['paper']}")
            print(f"     {pf['detail']}")
            for e in pf["emails"]:
                print(f"       {e}")
    
    # 论文风险总览
    print(f"\n{'─'*60}")
    print(f"  论文风险信号总览")
    print(f"{'─'*60}")
    
    for paper_name, signals in sorted(paper_signals.items(), key=lambda x: len(x[1]), reverse=True):
        if signals:
            n = len(signals)
            icon = "🔴" if n >= 3 else "🟡" if n >= 1 else "🟢"
            print(f"\n  {icon} {paper_name}: {n}个信号")
            for s in signals:
                print(f"     └─ {s}")
    
    # 无信号的论文
    clean = [p for p in paper_emails if p not in paper_signals or not paper_signals[p]]
    if clean:
        print(f"\n  🟢 无风险信号: {len(clean)}篇")
    
    if args.json:
        output = {
            "summary": summary,
            "cross_paper_emails": {e: papers for e, papers in cross_paper.items()},
            "pattern_findings": patterns,
            "cross_signals": cross_signals,
            "paper_signals": paper_signals,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n📄 JSON已保存: {args.json}")


if __name__ == "__main__":
    main()
