# Heaven's Net 更新日志

## v7.0 (2026-06-13)

### 新增

**1. 算术序列检测器 v7.0** (`scripts/arithmetic_sequence_detector_v7.py`)
- 无列名依赖 — 自动检测所有数值列
- 三层假阳性过滤：unique≥3 + 众数<60% + diff≠0
- 多格式支持：CSV / Excel / JSON
- ClinicalTrials.gov 集成（`--ctg` 标志）
- 终端友好摘要输出

**2. ClinicalTrials.gov 数据提取** (`scripts/ctg_api.py`)
- 搜索有结果的临床试验（按疾病/关键词）
- 提取结构化数值：outcome measures, p值, 效应量, 置信区间
- 批量和单个试验模式
- 已验证：190试验/21,421行/10,381组，0 API错误

**3. 依赖更新** (`requirements.txt`)
- 新增 `requests>=2.28.0`

### 改 vs v6.0

| 特性 | v6.0 | v7.0 |
|------|------|------|
| 列名要求 | 需特定格式 | 自动检测数值列 |
| 假阳性过滤 | 基础 | 三层强化过滤 |
| CT.gov 数据源 | 不支持 | `--ctg` 集成 |
| p值计算 | 原有逻辑 | 改进：考虑差值分布 |
| 质量过滤 | 无 | unique≥3, mode<60%, diff≠0 |

### 已知局限

1. **临床计数数据易假阳性** — 不良事件计数天然等差（0,1,2,3...），v7.0三层过滤已基本排除
2. **PDF表格提取** — 成功率 <10%，优先用 CT.gov API 或手动 CSV
3. **无监督大规模扫描不可行** — 原始数据99%不公开在可编程访问的位置

### 方法论来源

- `academic-critical-review` skill — GRADE/Cochrane RoB 2/统计欺诈模式
- 固定差值检测原理 — 在 738个 CT.gov 试验上验证了假阳性控制
