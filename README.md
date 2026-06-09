# Heaven's Net — 学术不端检测系统 v1.2

> 天网恢恢，疏而不漏。中英文双轨学术论文造假自动化检测系统。
> 覆盖数据统计反算、图片取证、文本分析、跨论文横向比对、三层次统计方法审计五大维度。

## 概述

Heaven's Net 从公开的学术不端案例中提炼检测方法论，固化为 18 个独立可运行的 Python 脚本，加上一个统一的 CLI 入口。

**核心设计理念：**
- **假阳性率优先于敏感性** — 宁可漏过，不可冤枉
- **不可推翻的证据链** — 每个发现必须有定位 + 观察 + 规则 + 严重性 + 替代解释
- **三层次统计审计** — 方法对不对 → 数据对不对 → 结论对不对
- **蒙特卡洛基线** — 所有算术检测器经 10,000 次随机模拟验证（FPR < 0.5%）
- **中英文双轨** — 同一套管线同时支持中文和英文论文

## 快速开始

### 安装依赖

```bash
pip install PyMuPDF Pillow openpyxl scipy numpy beautifulsoup4
```

### 审查一篇论文

```bash
# 全管线审查（数据+图片+文本+统计审计+逻辑）
python3 scripts/unified_review.py 论文.pdf

# 输出 Markdown 报告
python3 scripts/unified_review.py 论文.pdf --format markdown --output 报告.md

# 审查整个目录（多篇论文 + 跨论文比对）
python3 scripts/unified_review.py ~/论文目录/ --cross-file
```

### 单独运行统计方法审计

```bash
# 三层次统计方法审计（自动提取方法声明 → 检查适用性 → 反算P值）
python3 scripts/stat_method_auditor.py 论文.pdf

# 指定数据类型
python3 scripts/stat_method_auditor.py 论文.pdf --data-type continuous_normal_multi_groups
```

### 单独运行其他检测器

```bash
# 算术检测（固定差值/比例/等差数列/末位数字/Decimal前缀等）
python3 scripts/arithmetic_sequence_detector.py 数据.csv

# 图片快速初筛（aHash + 6方向旋转翻转检测）
python3 scripts/image_hash_screener.py 图片目录/
```

## 检测能力总览

### 一、三层次统计方法审计（🆕 核心能力）

| 层次 | 检查内容 | 示例问题 |
|:--:|------|------|
| **第1层** | 声称的方法是否正确？ | 计数数据用了 t 检验？重复测量数据用了单因素 ANOVA？ |
| **第2层** | 用他的方法反算，结果对不对？ | 论文报告 P<0.01，但用同样方法反算 P=0.97？ |
| **第3层** | 根据真实统计结果，结论是否成立？ | "各时间点均显著"但 24h 时间点实际不显著？ |

支持的统计方法：
独立样本 t 检验、配对 t 检验、单因素方差分析、重复测量方差分析、
卡方检验、Fisher 精确检验、Mann-Whitney U 检验、Wilcoxon 符号秩检验、
Kruskal-Wallis 检验、Pearson 相关、Spearman 等级相关、
Kaplan-Meier 生存分析、log-rank 检验、Cox 回归

### 二、数据统计检测（A1-A16）

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

### 三、图片取证检测

| 检测器 | 能力 |
|------|------|
| **aHash 初筛** | 6方向变体全量交叉比对 |
| **ELA（误差水平分析）** | 检测 JPEG 压缩不一致 → 图片合成痕迹 |
| **块重复检测** | 同一图片内完全相同的像素块 → 复制粘贴 |
| **背景突变检测** | 背景灰度/纹理突变 → 拼接痕迹 |
| **SIFT 特征点匹配** | 旋转/翻转不变的特征点匹配 |
| **Western Blot 条带互相关** | 条带级互相关分析 |
| **WB 泳道拼接检测** | 列强度剖面 z-score + 泳道哈希重复 |

### 四、中文专项检测

| 检测器 | 能力 |
|------|------|
| **中文文本异常** | P值语法矛盾、中文受虐短语 |
| **跨论文邮箱关联** | 同一邮箱出现在多篇论文中 |
| **作者异常检测** | 作者贡献与数据量不匹配 |

## 目录结构

```
heavens-net/
├── README.md
├── requirements.txt
├── .gitignore
└── scripts/
    ├── unified_review.py              ← CLI 主入口
    ├── stat_method_auditor.py          ← 🆕 三层次统计方法审计
    ├── arithmetic_sequence_detector.py ← A1-A16 算术检测
    ├── twelve_weapons.py               ← 统计反算 12 武器
    ├── monte_carlo_baseline.py         ← 蒙特卡洛基线
    ├── image_hash_screener.py          ← aHash 图片初筛
    ├── image_forensics_pipeline.py     ← ELA+块重复+背景突变
    ├── sift_duplicate_detection.py     ← SIFT 特征点匹配
    ├── analyze_wb_bands.py             ← WB 条带互相关
    ├── blot_gel_lane_audit.py          ← WB 泳道拼接
    ├── bulk_cross_paper_analysis.py    ← 跨论文图片比对
    ├── chinese_tortured_phrases.py     ← 中文文本异常
    ├── chinese_stat_parser.py          ← 中文统计声明解析
    ├── cross_paper_email_analyzer.py   ← 跨论文邮箱
    ├── author_anomaly_detector.py      ← 作者异常
    └── supplementary_data_fetcher.py   ← 补充数据抓取
```

## 审查流程（八步法 + 三层次统计审计）

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
⑤ 第2层：报告格式+算术模式（A1-A16）
    ↓
⑥ 第3层：aHash初筛→ELA+块重复+背景突变→WB泳道→WB条带→pHash跨图→SIFT
    ↓
⑦ 第4+5层：跨论文归因+外部验证
    ↓
⑧ 独立报告（证据等级 L1-L5）
```

## 证据等级

| 等级 | 含义 | 示例 |
|:--:|------|------|
| L1 | 轻微异常，可忽略 | 末位数字略偏 |
| L2 | 存疑，需关注 | 小数点前缀重复 >15% |
| L3 | 高度可疑 | 多列固定差值匹配 ≥50% |
| L4 | 极可能是造假 | Byte-identical 行重复 + ELA 异常 |
| L5 | 不可推翻的硬证据 | 两列数据逐字节相同 + 图片像素级重叠 |

## 限制与免责

- 所有检测结果为筛查证据，不是最终定论
- aHash 初筛对白底为主的图片会产生假阳性，需 ELA/SIFT 二次验证
- WB 泳道检测对所有图片运行，对非 WB 图会产生假阳性
- 三层次审计的第2层反算需要在独立模式下手动提供 data_blocks
- 本工具集供学术自查和研究用途，不替代期刊/机构的正式调查

## License

MIT
