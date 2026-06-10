# Heaven's Net — Academic Fraud Detection System v1.2

> 天网恢恢，疏而不漏。中英文双轨学术论文造假自动化检测系统。
> Heaven's net is vast, its mesh is fine, yet nothing slips through.

[English](#english) | [中文](#中文)

---

## English

Heaven's Net is an open-source, bilingual (Chinese/English) system for automated detection of scientific paper fabrication, image manipulation, and statistical misconduct. It extracts detection methodologies from public cases of academic fraud and solidifies them into 19 standalone Python scripts with a unified CLI entry point, combining **38+ detectors** across **five dimensions**.

### What it does

- **Statistical reverse-engineering** — reconstructs test statistics from reported data and compares with claimed p-values (3-level audit: method correctness → data validity → conclusion consistency)
- **Arithmetic pattern detection** — constant difference/ratio, last-digit bias, decimal prefix repetition, byte-identical blocks (16 weapons, Monte Carlo baseline: FPR < 0.5%)
- **Image forensics** — ELA, block duplication, background mutation, SIFT feature matching, Western Blot lane splicing, AI-generated image frequency-domain detection
- **Cross-paper analysis** — email association, author anomaly, pHash cross-comparison
- **Chinese-specific detection** — tortured phrases (machine translation artifacts), Chinese statistical reporting patterns

### Quick Start

```bash
# Install dependencies
pip install PyMuPDF Pillow openpyxl scipy numpy beautifulsoup4

# Full pipeline review of a single paper
python3 scripts/unified_review.py paper.pdf

# Output Markdown report
python3 scripts/unified_review.py paper.pdf --format markdown --output report.md

# Cross-file mode (batch review + cross-paper comparison)
python3 scripts/unified_review.py ~/papers/ --cross-file

# Statistical method audit only
python3 scripts/stat_method_auditor.py paper.pdf

# Arithmetic detection on raw CSV
python3 scripts/arithmetic_sequence_detector.py data.csv

# Image hash screening
python3 scripts/image_hash_screener.py images/
```

### Detection Capabilities

#### 1. Three-Level Statistical Method Audit (Core)

| Level | What it checks | Example problem |
|:-----:|----------------|-----------------|
| **Level 1** | Is the claimed statistical method correct? | Count data analyzed with t-test? Repeated measures treated as one-way ANOVA? |
| **Level 2** | When recalculating with the same method, do results match? | Paper reports P<0.01, but independent recalculation yields P=0.97 |
| **Level 3** | Do conclusions follow from the actual statistical results? | "All time points significant" but the 24h time point is not |

**Supported methods:** Independent t-test, paired t-test, one-way ANOVA, repeated measures ANOVA, chi-square test, Fisher's exact test, Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Pearson/Spearman correlation, Kaplan-Meier survival analysis, log-rank test, Cox regression.

#### 2. Arithmetic & Data Pattern Detectors (A1–A16)

| Weapon | Detection | Origin |
|:------:|-----------|--------|
| A1 | Fixed difference (constant gap between two columns) | Public cases |
| A2 | Fixed ratio | — |
| A3 | Linear relationship (a×col1 + b = col2) | — |
| A4 | Single-column arithmetic sequence | Public cases |
| A5 | Last-digit preference (0+5 concentration) | — |
| A6 | Decimal precision anomaly | — |
| A7 | Block-level repetition (groups of columns repeat together) | — |
| A8 | Noise-environment fabrication pattern | — |
| A9 | Byte-identical rows/blocks/columns | Published methodology |
| A10 | Constant shift detection | Public cases |
| A11 | Cross-group row copy-paste | — |
| A12 | Decimal prefix repetition | Published methodology |
| A13 | Arithmetic grid | — |
| A14 | Column copy with one cell changed | — |
| A15 | Statistical back-calculation (12 weapons) | — |
| A16 | Chinese statistical statement parsing | — |

#### 3. Image Forensics

| Detector | Capability |
|----------|------------|
| **aHash screening** | Full cross-comparison over 6 transformations |
| **ELA (Error Level Analysis)** | JPEG compression inconsistency → image compositing traces |
| **Block duplication** | Identical pixel blocks within an image → copy-move forgery |
| **Background mutation** | Sudden changes in background gray level/texture → splice marks |
| **SIFT feature matching** | Rotation/flip-invariant feature point matching |
| **Western Blot band cross-correlation** | Inter-band cross-correlation analysis |
| **WB lane splicing** | Column intensity profile z-score + lane hash repetition |
| **AI image frequency-domain** | Frequency-domain patterns characteristic of AI-generated figures |

#### 4. Chinese / Cross-Paper Detectors

| Detector | Capability |
|----------|------------|
| **Chinese text anomaly** | P-value grammatical contradictions, tortured phrases (machine translation artifacts) |
| **Cross-paper email correlation** | Same email address appearing across multiple papers |
| **Author anomaly** | Mismatch between author contributions and data volume |

### Audit Flow (8-Step + 3-Level Statistical Audit)

```
① Author verification → ② Data-layer judgment → ③ Section-by-section method reading
    │                                                    │
    │                                      🆕 3-Level Statistical Audit
    │                                      Level 1: Is the method correct?
    │                                      Level 2: Is the data correct?
    │                                      Level 3: Is the conclusion correct?
    ↓
④ Layer 1: Statistical reverse-engineering (15 weapons)
    ↓
⑤ Layer 2: Report format + arithmetic patterns (A1–A16)
    ↓
⑥ Layer 3: aHash screening → ELA + block duplication + background mutation → WB lanes → WB bands → pHash cross-paper → SIFT
    ↓
⑦ Layers 4+5: Cross-paper attribution + external validation
    ↓
⑧ Independent report (Evidence Levels L1–L5)
```

### Evidence Levels

| Level | Meaning | Example |
|:-----:|---------|---------|
| L1 | Minor anomaly, ignorable | Slight last-digit skew |
| L2 | Suspicious, needs attention | Decimal prefix repetition >15% |
| L3 | Highly suspicious | Multi-column fixed-difference match ≥50% |
| L4 | Likely fabrication | Byte-identical row duplication + ELA anomaly |
| L5 | Irrefutable hard evidence | Two columns byte-for-byte identical + pixel-level image overlap |

### False Positive Rates

*Validated on 30 PLOS ONE 2024 papers as negative control.*

| Detector | FPR | Status |
|----------|:---:|:------:|
| Fixed difference/ratio (Monte Carlo) | 0% | ✅ Trusted |
| ELA, block duplication, background mutation | 0% | ✅ Trusted |
| AI image frequency-domain detection | 0% | ✅ Trusted |
| Tortured phrases, WB band correlation | 0% | ✅ Trusted |
| Cross-table consistency | 3.3% | ✅ Low |
| aHash image screening | 6.7% | ⚠️ Moderate |
| SIFT feature matching | 60% | 🚨 Manual review required |
| WB lane splicing | 73% | 🚨 Manual review required |

### Key Design Principles

- **False positive rate over sensitivity** — better to miss than to falsely accuse
- **Monte Carlo baseline** — all arithmetic detectors validated with 10,000 random simulations (FPR < 0.5%)
- **Negative control baseline** — validated on 30 clean papers to establish real-world FPR
- **Irrefutable evidence chain** — every finding includes location, observation, rule, severity, and alternative explanation
- **MIT License** — free for academic and commercial use

### Directory Structure

```
heavens-net/
├── README.md
├── requirements.txt
├── .gitignore
└── scripts/
    ├── unified_review.py              ← Main CLI entry point
    ├── stat_method_auditor.py          ← 3-level statistical method audit
    ├── arithmetic_sequence_detector.py ← A1–A16 arithmetic detection
    ├── twelve_weapons.py               ← Statistical back-calculation (12 weapons)
    ├── monte_carlo_baseline.py         ← Monte Carlo baseline
    ├── image_hash_screener.py          ← aHash image screening
    ├── image_forensics_pipeline.py     ← ELA + block duplication + background mutation
    ├── sift_duplicate_detection.py     ← SIFT feature matching
    ├── analyze_wb_bands.py             ← WB band cross-correlation
    ├── blot_gel_lane_audit.py          ← WB lane splicing
    ├── bulk_cross_paper_analysis.py    ← Cross-paper image comparison
    ├── chinese_tortured_phrases.py     ← Chinese text anomaly
    ├── chinese_stat_parser.py          ← Chinese statistical statement parsing
    ├── cross_paper_email_analyzer.py   ← Cross-paper email correlation
    ├── author_anomaly_detector.py      ← Author anomaly
    └── supplementary_data_fetcher.py   ← Supplementary data fetching
```

### Limitations & Disclaimer

- All detection results are screening evidence, not final verdicts
- aHash screening produces false positives on white-background images; always verify with ELA/SIFT
- WB lane detection runs on all images and produces false positives on non-WB figures
- Level 2 statistical recalculation requires manually providing data_blocks in standalone mode
- This toolkit is for academic self-auditing and research purposes; it does not replace formal journal/institutional investigations

---

## 中文

Heaven's Net（天网恢恢，疏而不漏）从公开的学术不端案例中提炼检测方法论，固化为 19 个独立可运行的 Python 脚本，加上一个统一的 CLI 入口，覆盖**数据统计反算、图片取证、文本分析、跨论文横向比对、三层次统计方法审计**五大维度，共 38+ 检测器。

**核心设计理念：** 假阳性率优先于敏感性（宁可漏过，不可冤枉） · 不可推翻的证据链 · 蒙特卡洛基线验证（FPR < 0.5%） · 中英文双轨 · MIT 开源

### 快速开始

```bash
pip install PyMuPDF Pillow openpyxl scipy numpy beautifulsoup4

python3 scripts/unified_review.py 论文.pdf
python3 scripts/unified_review.py 论文.pdf --format markdown --output 报告.md
python3 scripts/unified_review.py ~/论文目录/ --cross-file
python3 scripts/stat_method_auditor.py 论文.pdf
python3 scripts/arithmetic_sequence_detector.py 数据.csv
python3 scripts/image_hash_screener.py 图片目录/
```

### 检测能力总览

**一、三层次统计方法审计（核心）**

| 层次 | 检查内容 | 示例问题 |
|:--:|------|------|
| 第1层 | 声称的方法是否正确？ | 计数数据用了 t 检验？重复测量数据用了单因素 ANOVA？ |
| 第2层 | 用同样方法反算，结果对不对？ | 论文报告 P<0.01，但独立反算 P=0.97 |
| 第3层 | 结论是否基于真实统计结果？ | "各时间点均显著"但 24h 时间点实际不显著 |

**二、数据统计检测（A1–A16）**

| 武器 | 检测内容 | 来源 |
|:--:|------|------|
| A1 | 固定差值（两列差值恒定） | 公开案例 |
| A2 | 固定比值 | — |
| A3 | 线性关系（a×列1+b=列2） | — |
| A4 | 单列等差数列 | 公开案例 |
| A5 | 末位数字偏好（0+5集中度） | — |
| A6 | 小数点精度异常 | — |
| A7 | 块级重复（多列一起重复） | — |
| A8 | 噪声环境编造模式 | — |
| A9 | Byte-identical 行/块/列重复 | 公开方法论 |
| A10 | 常数平移检测 | 公开案例 |
| A11 | 跨组行级复制粘贴 | — |
| A12 | Decimal 前缀重复 | 公开方法论 |
| A13 | 等差网格 | — |
| A14 | 列复制改1格 | — |
| A15 | 统计反算（12武器） | — |
| A16 | 中文统计声明解析 | — |

**三、图片取证**

| 检测器 | 能力 |
|------|------|
| aHash 初筛 | 6方向变体全量交叉比对 |
| ELA（误差水平分析） | JPEG 压缩不一致 → 图片合成痕迹 |
| 块重复检测 | 同一图片内完全相同的像素块 → 复制粘贴 |
| 背景突变检测 | 背景灰度/纹理突变 → 拼接痕迹 |
| SIFT 特征点匹配 | 旋转/翻转不变的特征点匹配 |
| WB 条带互相关 | 条带级互相关分析 |
| WB 泳道拼接 | 列强度剖面 z-score + 泳道哈希重复 |

**四、中文专项 / 跨论文检测**

| 检测器 | 能力 |
|------|------|
| 中文文本异常 | P值语法矛盾、中文受虐短语 |
| 跨论文邮箱关联 | 同一邮箱出现在多篇论文中 |
| 作者异常检测 | 作者贡献与数据量不匹配 |

### 审查流程（八步法 + 三层次统计审计）

```
① 身份确认 → ② 数据层判断 → ③ 逐段读方法
    │                              │
    │                    🆕 三层次统计审计
    │                    第1层: 方法对不对？
    │                    第2层: 数据对不对？
    │                    第3层: 结论对不对？
    ↓
④ 第1层：统计反算（15武器）
    ↓
⑤ 第2层：报告格式+算术模式（A1–A16）
    ↓
⑥ 第3层：aHash初筛→ELA+块重复+背景突变→WB泳道→WB条带→pHash跨图→SIFT
    ↓
⑦ 第4+5层：跨论文归因+外部验证
    ↓
⑧ 独立报告（证据等级 L1–L5）
```

### 证据等级

| 等级 | 含义 | 示例 |
|:--:|------|------|
| L1 | 轻微异常，可忽略 | 末位数字略偏 |
| L2 | 存疑，需关注 | 小数点前缀重复 >15% |
| L3 | 高度可疑 | 多列固定差值匹配 ≥50% |
| L4 | 极可能是造假 | Byte-identical 行重复 + ELA 异常 |
| L5 | 不可推翻的硬证据 | 两列数据逐字节相同 + 图片像素级重叠 |

### 限制与免责

- 所有检测结果为筛查证据，不是最终定论
- aHash 初筛对白底为主的图片会产生假阳性，需 ELA/SIFT 二次验证
- WB 泳道检测对所有图片运行，对非 WB 图会产生假阳性
- 第2层统计反算需要在独立模式下手动提供 data_blocks
- 本工具集供学术自查和研究用途，不替代期刊/机构的正式调查

## License

MIT
