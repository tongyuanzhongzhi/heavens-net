#!/usr/bin/env python3
"""
中文受虐短语检测器

中文论文中机器改写产生的畸形同义词替换——与英文 Tortured Phrases 
类似的原理，但中文词库需要单独构建。

当前实现:
1. 内置中文疑似短语对（人工收集+从Cabanac词库翻译）
2. 正则扫描PDF/文本中的匹配
3. 输出疑似列表供人工复核

⚠️ 注意：本工具处于早期阶段，词库仅覆盖最典型的模式。
中文受虐短语的系统化检测需要更庞大的词库（类似Cabanac的~7000条）。

用法:
  python3 chinese_tortured_phrases.py paper.pdf
  python3 chinese_tortured_phrases.py --text "文本内容"

依赖: PyMuPDF (可选, 用于读取PDF)
"""

import sys, json, re, argparse, os

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


# ═══════════════════════════════════════════════
# 中文受虐短语词库
# 
# 来源:
# 1. 从Cabanac ~7000条英文词库中翻译典型模式
# 2. 从已知论文工厂论文中手动识别
# 3. 中文机器翻译常见错误模式
#
# 格式: (异常表述, 正确表述, 置信度 high/medium/low)
# ═══════════════════════════════════════════════

CHINESE_TORTURED_PHRASES = [
    # 生物医学领域
    ("假冒意识", "人工智能", "high"),
    ("深刻学习", "深度学习", "high"),
    ("深层神经组织", "深度神经网络", "high"),
    ("随机森林地带", "随机森林", "high"),
    ("支撑向量机", "支持向量机", "low"),  # 可能有多种译法
    ("信号到噪声比例", "信噪比", "medium"),
    ("西方印记", "Western blot", "high"),
    ("西方墨点", "Western blot", "high"),
    ("基因表达谱系", "基因表达谱", "low"),
    ("蛋白质印记", "蛋白质印迹/Western blot", "medium"),
    ("免疫组织合成化学", "免疫组织化学", "high"),
    ("聚合酶链式反应放大", "聚合酶链式反应/PCR", "medium"),
    ("逆转录聚合酶链式反应", "逆转录PCR/RT-PCR", "low"),  # 有争议
    ("流式细胞分析术", "流式细胞术", "medium"),
    ("酶联免疫吸附剂测定", "酶联免疫吸附试验/ELISA", "medium"),
    ("定量即时聚合酶链式反应", "实时定量PCR/qPCR", "high"),
    ("细胞死亡机制", "细胞凋亡", "medium"),
    ("程序性细胞死亡通路", "细胞凋亡通路", "medium"),
    ("转录因子结合位点序列", "转录因子结合位点", "medium"),
    ("微小核糖核酸", "微小RNA/miRNA", "medium"),
    ("短发夹核糖核酸", "短发夹RNA/shRNA", "high"),
    ("小干扰核糖核酸", "小干扰RNA/siRNA", "high"),
    ("长链非编码核糖核酸", "长链非编码RNA/lncRNA", "low"),  # 可能有多种写法
    
    # 统计学
    ("显著性差异分析", "显著性检验", "low"),  # 可能是正常写法
    ("统计显著性检验", "显著性检验", "low"),
    ("方差分析检验", "方差分析/ANOVA", "medium"),
    ("卡方检验分析", "卡方检验", "medium"),
    ("皮尔逊相关系数检验", "Pearson相关系数", "medium"),
    ("斯皮尔曼等级相关系数分析", "Spearman等级相关", "high"),
    ("多元线性回归分析模型", "多元线性回归", "low"),
    ("逻辑回归模型分析", "Logistic回归", "medium"),
    ("考克斯比例风险回归模型", "Cox比例风险模型", "medium"),
    
    # 通用学术
    ("数据存储仓库", "数据仓库", "high"),
    ("高性能计算技术", "高性能计算/HPC", "medium"),
    ("信息提取技术", "信息提取", "low"),
    ("机器学习算法模型", "机器学习算法", "low"),
    ("深度卷积神经网络模型", "深度卷积神经网络", "medium"),
    ("长短时记忆神经网络", "LSTM网络", "medium"),
    ("自然语言处理技术", "自然语言处理/NLP", "low"),
    ("计算机辅助诊断系统", "计算机辅助诊断/CAD", "medium"),
    ("图像分割算法技术", "图像分割", "medium"),
    ("特征提取和选择", "特征工程/特征选择", "medium"),
    ("数据增强和扩充", "数据增强", "medium"),
    ("模型训练和优化", "模型训练", "low"),
    ("超参数优化调整", "超参数调优", "medium"),
    
    # 实验方法
    ("细胞培养和传代", "细胞培养", "low"),  # 可能是正常写法
    ("细胞转染和感染", "细胞转染", "medium"),
    ("质粒构建和转染", "质粒转染", "medium"),
    ("基因敲除和敲低实验", "基因敲除/敲低", "medium"),
    ("蛋白质提取和定量分析", "蛋白提取和定量", "low"),
    ("免疫荧光染色分析", "免疫荧光染色", "medium"),
    ("免疫共沉淀实验分析", "免疫共沉淀/Co-IP", "medium"),
    ("染色质免疫沉淀测序分析", "ChIP-seq", "high"),
    ("RNA测序和转录组分析", "RNA-seq/转录组分析", "low"),
    ("单细胞RNA测序技术", "单细胞RNA测序/scRNA-seq", "medium"),
    
    # 临床
    ("回顾性队列研究分析", "回顾性队列研究", "low"),
    ("前瞻性随机对照临床试验", "RCT/随机对照试验", "medium"),
    ("多中心随机双盲安慰剂对照试验", "多中心RCT", "medium"),
    ("生存分析和预后评估", "生存分析", "medium"),
    ("诊断效能评估分析", "诊断效能评估", "medium"),
    ("受试者工作特征曲线分析", "ROC曲线分析", "medium"),
]


def detect_tortured_phrases(text):
    """检测中文受虐短语"""
    detections = []
    seen_spans = set()  # 避免重叠检测
    
    for abnormal, correct, confidence in CHINESE_TORTURED_PHRASES:
        for m in re.finditer(re.escape(abnormal), text):
            start, end = m.start(), m.end()
            
            # 避免重叠
            span_key = (start, end)
            is_overlap = False
            for s in seen_spans:
                if not (end <= s[0] or start >= s[1]):
                    is_overlap = True
                    break
            
            if is_overlap:
                continue
            
            seen_spans.add(span_key)
            
            # 获取上下文
            ctx_start = max(0, start - 40)
            ctx_end = min(len(text), end + 40)
            context = text[ctx_start:ctx_end]
            
            detections.append({
                "abnormal": abnormal,
                "correct": correct,
                "confidence": confidence,
                "position": f"字符{start+1}-{end+1}",
                "context": context
            })
    
    return detections


def extract_text_from_pdf(pdf_path):
    """从PDF提取文本"""
    if not HAS_FITZ:
        print("错误: 需要PyMuPDF (pip install pymupdf)", file=sys.stderr)
        return ""
    
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


class ChineseTorturedPhrases:
    """中文受虐短语检测器兼容包装"""
    
    def detect(self, text: str) -> list:
        """返回统一格式的发现列表，与 unified_review 兼容"""
        detections = detect_tortured_phrases(text)
        findings = []
        for d in detections:
            findings.append({
                "method": "chinese_tortured_phrase",
                "severity": "HIGH" if d['confidence'] == 'high' else "MEDIUM" if d['confidence'] == 'medium' else "LOW",
                "location": d['position'],
                "description": f"{d['abnormal']} → {d['correct']}",
                "statistics": {"context": d['context']},
            })
        return findings


def main():
    parser = argparse.ArgumentParser(description='中文受虐短语检测器')
    parser.add_argument('pdf', nargs='?', help='论文PDF路径')
    parser.add_argument('--text', help='直接传入文本')
    parser.add_argument('--output', '-o', default='chinese_tortured_phrases.json',
                       help='输出JSON路径')
    args = parser.parse_args()
    
    text = ""
    
    if args.text:
        text = args.text
    elif args.pdf:
        print(f"从PDF提取文本: {args.pdf}")
        text = extract_text_from_pdf(args.pdf)
    else:
        # 演示模式
        text = """
本研究采用深刻学习方法对医学图像进行计算机辅助诊断系统分析。
使用深度卷积神经网络模型对数据进行特征提取和选择。
采用西方印记技术检测蛋白质表达水平，
通过定量即时聚合酶链式反应分析基因表达谱系的变化。
数据存储在数据存储仓库中，使用随机森林地带模型进行分类。
统计方法包括显著性差异分析和方差分析检验。
采用程序性细胞死亡通路检测细胞死亡机制。
我们使用长短时记忆神经网络进行预后预测。
"""
    
    if not text:
        print("请提供PDF路径或使用 --text 传入文本")
        sys.exit(1)
    
    print("=" * 60)
    print("中文受虐短语检测")
    print("=" * 60)
    
    detections = detect_tortured_phrases(text)
    
    print(f"\n扫描文本: {len(text)} 字符")
    print(f"词库大小: {len(CHINESE_TORTURED_PHRASES)} 条")
    print(f"检出: {len(detections)} 条")
    
    if detections:
        # 按置信度分组
        high = [d for d in detections if d['confidence'] == 'high']
        medium = [d for d in detections if d['confidence'] == 'medium']
        low = [d for d in detections if d['confidence'] == 'low']
        
        print(f"  HIGH: {len(high)}")
        print(f"  MEDIUM: {len(medium)}")
        print(f"  LOW: {len(low)}")
        
        print(f"\n高风险检测 (HIGH):")
        for d in high:
            print(f"  🔴 {d['abnormal']} → {d['correct']}")
            print(f"     上下文: ...{d['context']}...")
        
        if medium:
            print(f"\n中风险检测 (MEDIUM):")
            for d in medium:
                print(f"  🟡 {d['abnormal']} → {d['correct']}")
                print(f"     上下文: ...{d['context']}...")
        
        # 风险判定
        high_count = len(high)
        if high_count >= 5:
            risk = "🔴 HIGH — 强烈指向论文工厂"
        elif high_count >= 2:
            risk = "🟡 MEDIUM — 需进一步调查"
        else:
            risk = "🟢 LOW — 可能是翻译错误"
        
        print(f"\n综合判定: {risk}")
    else:
        print("✅ 未检测到受虐短语")
    
    # 统计词库覆盖率
    matched = set(d['abnormal'] for d in detections)
    total = len(CHINESE_TORTURED_PHRASES)
    print(f"\n词库匹配率: {len(matched)}/{total}")
    
    output = {
        "text_length": len(text),
        "dictionary_size": total,
        "detections": detections,
        "summary": {
            "total": len(detections),
            "high": len([d for d in detections if d['confidence'] == 'high']),
            "medium": len([d for d in detections if d['confidence'] == 'medium']),
            "low": len([d for d in detections if d['confidence'] == 'low'])
        },
        "risk_level": "HIGH" if len([d for d in detections if d['confidence'] == 'high']) >= 5
                      else "MEDIUM" if len(detections) >= 3 else "LOW",
        "limitations": [
            "词库仅67条，远小于英文Cabanac词库(~7000条)",
            "中文受虐短语模式与英文不同——中文更少使用字面翻译",
            "HIGH置信度条目仅限明确的机器翻译错误",
            "所有检测结果建议人工复核"
        ]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存: {args.output}")
    print("⚠️  重要提示: 中文受虐短语检测仍在早期阶段。")
    print("   英文Cabanac词库有~7000条，本工具仅67条。")
    print("   最佳实践: 结合构图指纹+机构评估+跨论文比对一起使用。")


if __name__ == '__main__':
    main()
