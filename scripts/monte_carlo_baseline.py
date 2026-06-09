#!/usr/bin/env python3
"""
蒙特卡洛模拟：为算术检测器15项（A1-A15）建立阴性对照基线。

方法：生成 n_sim=1000 组完全随机的正态分布数据，模拟清洁 RCT 的 Table 1 基线特征，
每组跑 ArithmeticSequenceDetector 的 A1-A15（排除需要图片的检测），
统计每个检测器在纯随机数据中的假阳性触发次数。

最终产出：
- 每个检测器的假阳性触发率（false positive rate, FPR）
- 高置信度结论：「p < 0.001」级别 —— 如果 1000 次模拟中 0 次触发

用法：python monte_carlo_baseline.py [--n-sim 1000] [--n-per-group 60] [--n-groups 4]
"""

import csv
import json
import math
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

# 确保能导入同目录的 arithmetic_sequence_detector
sys.path.insert(0, str(Path(__file__).parent))
from arithmetic_sequence_detector import ArithmeticSequenceDetector


def generate_rct_baseline_table(n_per_group=60, n_groups=4, n_variables=8):
    """
    生成一个模拟的 RCT Table 1 基线特征表。
    
    参数:
        n_per_group: 每组样本数（典型 RCT: 30-100）
        n_groups: 组数（典型: 2-4）
        n_variables: 基线变量数
    
    返回:
        (headers, rows) — 类似 CSV 格式，可直接喂给 ArithmeticSequenceDetector
    """
    # 表头: Group1_Age, Group1_BMI, ..., Group2_Age, Group2_BMI, ...
    headers = []
    for g in range(n_groups):
        for v in range(n_variables):
            headers.append(f"G{g+1}_V{v+1}")
    
    # 不同变量的真实分布参数（参考 SPRINTT 试验的基线数据范围）
    variable_params = [
        {"name": "Age", "mean": 78.5, "sd": 5.2, "min": 70, "max": 95},       # V1
        {"name": "BMI", "mean": 28.3, "sd": 4.8, "min": 18, "max": 45},         # V2
        {"name": "SPPB", "mean": 8.2, "sd": 1.5, "min": 3, "max": 12},          # V3
        {"name": "GaitSpeed", "mean": 0.85, "sd": 0.18, "min": 0.4, "max": 1.3},# V4
        {"name": "CRP", "mean": 3.5, "sd": 2.8, "min": 0.3, "max": 20},         # V5
        {"name": "Hb", "mean": 12.8, "sd": 1.5, "min": 9, "max": 17},           # V6
        {"name": "SBP", "mean": 135, "sd": 18, "min": 100, "max": 180},         # V7
        {"name": "Creatinine", "mean": 0.95, "sd": 0.25, "min": 0.4, "max": 1.8},# V8
    ][:n_variables]
    
    # 生成数据矩阵: [group][variable][subject]
    data = []
    for g in range(n_groups):
        group_data = []
        for v, params in enumerate(variable_params):
            col = []
            for _ in range(n_per_group):
                val = random.gauss(params["mean"], params["sd"])
                # 截尾到合理范围，然后四舍五入保持自然数据的外观
                val = max(params["min"], min(params["max"], val))
                # 根据测量精度四舍五入（所有连续变量 round 到 1 位小数）
                # 年龄也是 1 位小数——真实 RCT 的 Table 1 中年龄常精确到 0.1
                if params["sd"] < 0.5:
                    val = round(val, 2)  # CRP 等精密检测
                else:
                    val = round(val, 1)  # 年龄、BMI、SPPB、Hb、SBP、Creatinine
                col.append(val)
            group_data.append(col)
        data.append(group_data)
    
    # 转置为行格式（CSV格式）
    rows = []
    for s in range(n_per_group):
        row = []
        for g in range(n_groups):
            for v in range(n_variables):
                row.append(data[g][v][s])
        rows.append([str(x) for x in row])
    
    return headers, rows


def run_single_trial(detector, trial_id, n_per_group=60, n_groups=4, n_variables=8):
    """
    单次试验：生成一组随机数据 → 写入临时 CSV → 跑 A1-A15 检测器。
    返回触发的检测类型列表。
    """
    headers, rows = generate_rct_baseline_table(n_per_group, n_groups, n_variables)
    
    # 写入临时CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        tmp_path = f.name
    
    try:
        findings = detector.detect_from_file(tmp_path)
        triggered = [f["type"] for f in findings if "error" not in f]
        return {
            "trial": trial_id,
            "triggered_types": triggered,
            "triggered_count": len(triggered),
            "all_findings": findings
        }
    finally:
        Path(tmp_path).unlink()


def run_monte_carlo(n_sim=1000, n_per_group=60, n_groups=4, n_variables=8):
    """
    主函数：运行 n_sim 次蒙特卡洛试验，汇总假阳性率。
    """
    detector = ArithmeticSequenceDetector()
    results = []
    type_counter = Counter()
    
    print(f"开始蒙特卡洛模拟: {n_sim} 次试验")
    print(f"每组 {n_per_group} 样本, {n_groups} 个组, {n_variables} 个变量")
    print(f"总数据点: {n_sim} × {n_groups} × {n_variables} × {n_per_group} = {n_sim * n_groups * n_variables * n_per_group}")
    print()
    
    for trial in range(1, n_sim + 1):
        result = run_single_trial(detector, trial, n_per_group, n_groups, n_variables)
        results.append(result)
        
        for t in result["triggered_types"]:
            type_counter[t] += 1
        
        if trial % 100 == 0:
            print(f"  进度: {trial}/{n_sim}... 累计假阳性触发 = {sum(type_counter.values())}")
    
    print(f"\n完成 {n_sim} 次试验。")
    print(f"累计假阳性触发总次数: {sum(type_counter.values())}")
    
    return results, type_counter


def analyze_results(results, type_counter, n_sim):
    """
    分析结果，按照 unified-fraud-detection 框架中的 A1-A15 检测器分类输出。
    """
    # 检测器名称 → 框架编号 映射
    detector_map = {
        "ARITHMETIC_SEQUENCE": "A1 等差数列（单列内）",
        "FIXED_DIFFERENCE": "A2 固定差值（列间）",
        "DIGIT_PREFERENCE": "A3 末位数字偏好",
        "ARITHMETIC_GRID": "A10 等差网格",
        "CONSTANT_SHIFT": "A11 常数平移",
        "BYTE_IDENTICAL_ROWS": "A4 数值重复率 / A6 行级哈希重复",
        "BYTE_IDENTICAL_BLOCK": "A5 完全相同行 / A7 块级重复",
        "BYTE_IDENTICAL_COLUMNS": "A8 Byte-identical列",
        "BYTE_IDENTICAL_NEAR_COLUMNS": "A9 列复制改1格",
        "CROSS_GROUP_ROW_DUPLICATION": "A12 跨组行级复制粘贴",
        "NOISE_CORRELATION": "噪声相关性（额外检测）",
        "DECIMAL_PRECISION_ANOMALY": "小数点精度异常（额外检测）",
        "BLOCK_DUPLICATION": "块重复（额外检测）",
        "LINEAR_RELATION": "线性关系（额外检测）",
        "FIXED_RATIO": "固定比值（额外检测）",
    }
    
    # 框架 A1-A15 覆盖的所有检测类型
    a1_a15_types = [
        "ARITHMETIC_SEQUENCE",   # A1
        "FIXED_DIFFERENCE",      # A2
        "DIGIT_PREFERENCE",      # A3
        "BLOCK_DUPLICATION",     # A4/A5/A7 相关
        "BYTE_IDENTICAL_ROWS",   # A6
        "BYTE_IDENTICAL_BLOCK",  # A7
        "BYTE_IDENTICAL_COLUMNS",# A8
        "BYTE_IDENTICAL_NEAR_COLUMNS", # A9
        "ARITHMETIC_GRID",       # A10
        "CONSTANT_SHIFT",        # A11
        "CROSS_GROUP_ROW_DUPLICATION", # A12
    ]
    
    print("\n" + "="*70)
    print("蒙特卡洛模拟结果：算术检测器阴性对照基线")
    print(f"试验参数: n_sim={n_sim}, n_per_group=60, n_groups=4, n_variables=8")
    print("="*70)
    
    fpr_data = {}
    all_triggered = set(type_counter.keys())
    
    for dtype in a1_a15_types:
        count = type_counter.get(dtype, 0)
        fpr = count / n_sim
        label = detector_map.get(dtype, dtype)
        status = "✅ 0假阳性" if count == 0 else f"⚠️ {count}次 (FPR={fpr:.4f})"
        
        if count == 0:
            confidence = "p < 0.001（高置信度：假阳性率 < 0.1%）"
        elif fpr < 0.01:
            confidence = f"p ≈ {fpr:.4f}（低假阳性率）"
        else:
            confidence = f"p = {fpr:.4f}（需要阈值调优）"
        
        fpr_data[dtype] = {"count": count, "fpr": fpr}
        
        if count > 0:
            # 查看具体触发细节
            triggered_trials = [r for r in results if dtype in r["triggered_types"]]
            sample_finding = None
            for r in triggered_trials:
                for f in r["all_findings"]:
                    if f["type"] == dtype:
                        sample_finding = f["description"]
                        break
                if sample_finding:
                    break
            print(f"\n  {label}: {status}")
            print(f"      置信度: {confidence}")
            if sample_finding:
                print(f"      示例: {sample_finding[:100]}")
        else:
            print(f"\n  {label}: {status}")
            print(f"      置信度: {confidence}")
    
    # 未触发的检测器（理论上不可能在随机数据中触发）
    never_triggered = [d for d in a1_a15_types if d not in all_triggered]
    if never_triggered:
        print(f"\n--- 以下检测器在 {n_sim} 次模拟中从未触发 ---")
        for d in never_triggered:
            print(f"  {detector_map.get(d, d)}: 0/{n_sim}")
        print(f"\n  结论: 这些检测器的假阳性率 < 1/{n_sim} = {1/n_sim:.4f}")
        print(f"  在随机正态分布数据中，这些信号不会随机出现。")
    
    # 汇总
    total_false_positives = sum(type_counter.values())
    overall_fpr = total_false_positives / (n_sim * len(a1_a15_types))
    
    print(f"\n" + "="*70)
    print(f"汇总")
    print(f"  总试验次数: {n_sim}")
    print(f"  算术检测器数: {len(a1_a15_types)}")
    print(f"  总假阳性触发: {total_false_positives}")
    print(f"  整体 FPR: {overall_fpr:.6f}")
    print("="*70)
    
    return fpr_data


def save_baseline_report(fpr_data, n_sim, n_per_group, n_groups, n_variables, output_path):
    """将基线报告保存为 Markdown 文件，可直接写入 unified-fraud-detection skill"""
    
    detect_names = {
        "ARITHMETIC_SEQUENCE": "A1",
        "FIXED_DIFFERENCE": "A2",
        "DIGIT_PREFERENCE": "A3",
        "BLOCK_DUPLICATION": "A4/A5/A7",
        "BYTE_IDENTICAL_ROWS": "A6",
        "BYTE_IDENTICAL_BLOCK": "A7",
        "BYTE_IDENTICAL_COLUMNS": "A8",
        "BYTE_IDENTICAL_NEAR_COLUMNS": "A9",
        "ARITHMETIC_GRID": "A10",
        "CONSTANT_SHIFT": "A11",
        "CROSS_GROUP_ROW_DUPLICATION": "A12",
    }
    
    lines = []
    lines.append("# 算术检测器阴性对照基线（蒙特卡洛模拟）")
    lines.append("")
    lines.append(f"**生成日期:** 2026-06-06")
    lines.append(f"**生成方法:** 蒙特卡洛模拟 — {n_sim} 组完全随机的正态分布数据（模拟清洁 RCT 的 Table 1 基线特征）")
    lines.append("")
    lines.append("## 模拟参数")
    lines.append("")
    lines.append(f"- 模拟次数: {n_sim}")
    lines.append(f"- 每组样本数: {n_per_group}（对标 NEJM 等大型 RCT 的每组样本量）")
    lines.append(f"- 组数: {n_groups}（对标多臂 RCT）")
    lines.append(f"- 变量数: {n_variables}（对标 Table 1 的基线特征变量数）")
    lines.append(f"- 总数据点: {n_sim} × {n_groups} × {n_variables} × {n_per_group} = {n_sim * n_groups * n_variables * n_per_group}")
    lines.append(f"- 数据分布: 独立正态分布（每个变量使用真实 RCT 的均值和标准差参数）")
    lines.append("")
    lines.append("## 结果：每个检测器的假阳性触发次数")
    lines.append("")
    lines.append("| 检测器 | 编号 | 假阳性次数 | FPR | 置信度 |")
    lines.append("|--------|------|:---------:|:---:|--------|")
    
    all_zero = True
    for dtype, name in detect_names.items():
        info = fpr_data.get(dtype, {"count": 0, "fpr": 0.0})
        count = info["count"]
        fpr_val = info["fpr"]
        if count == 0:
            conf = f"p < {1/n_sim:.4f}"
            lines.append(f"| {dtype} | {name} | 0 | < {1/n_sim:.4f} | ✅ 高置信度静默 |")
        else:
            all_zero = False
            conf = f"p = {fpr_val:.4f}"
            lines.append(f"| {dtype} | {name} | {count} | {fpr_val:.4f} | ⚠️ {conf} |")
    
    lines.append("")
    
    if all_zero:
        lines.append(f"**结论: 所有算术检测器在 {n_sim} 次随机数据的模拟中均为零假阳性。**")
        lines.append(f"")
        lines.append(f"这意味着:")
        lines.append(f"- 等差数列/固定差值/常数平移/等差网格等模式在纯随机正态分布中不会自然出现")
        lines.append(f"- 任意一个算术检测器的假阳性率 < {1/n_sim}")
        lines.append(f"- 当 A1-A12 在论文中触发时，可高置信度排除「随机偶然」的替代解释")
    else:
        lines.append(f"**注意: {len([d for d, i in fpr_data.items() if i['count'] > 0])} 个检测器出现非零假阳性。**")
        lines.append(f"这些检测器在作为独立定罪证据时需要更谨慎——建议仅在配合其他层独立信号时使用。")
    
    lines.append("")
    lines.append("## 方法说明")
    lines.append("")
    lines.append("每轮模拟生成的数据是：")
    lines.append(f"1. 从真实 RCT（如 SPRINTT 试验）的参数中采样：均值、标准差、合理范围")
    lines.append(f"2. 使用 Python `random.gauss()` 生成独立正态分布数据")
    lines.append(f"3. 按真实 Table 1 的格式构造 CSV（每组 {n_variables} 列，每列 {n_per_group} 行）")
    lines.append(f"4. 将 CSV 喂给 ArithmeticSequenceDetector，记录所有触发信号")
    lines.append(f"5. 重复 {n_sim} 次后统计假阳性率")
    lines.append("")
    lines.append("**局限性（诚实标注）：**")
    lines.append("- 蒙特卡洛模拟只能确定「纯随机数据中不会触发」，不能完全替代真实清洁 RCT 的原始数据基线")
    lines.append("- 真实数据可能存在微小的非随机结构（如年龄的整数化、实验室检测的批次效应），但理论上这些不会产生精确的等差数列/固定差值")
    lines.append("- 如果某个检测器在真实清洁论文中从未触发且在模拟中也从未触发，我们可以同时从两个维度确认其可靠性")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='蒙特卡洛模拟：算术检测器阴性对照基线')
    parser.add_argument('--n-sim', type=int, default=1000, help='模拟次数 (默认: 1000)')
    parser.add_argument('--n-per-group', type=int, default=60, help='每组样本数 (默认: 60)')
    parser.add_argument('--n-groups', type=int, default=4, help='组数 (默认: 4)')
    parser.add_argument('--n-variables', type=int, default=8, help='变量数 (默认: 8)')
    parser.add_argument('--output', type=str, default=None, help='输出Markdown文件路径')
    parser.add_argument('--json', action='store_true', help='额外输出JSON格式')
    args = parser.parse_args()
    
    # 运行模拟
    results, type_counter = run_monte_carlo(
        n_sim=args.n_sim,
        n_per_group=args.n_per_group,
        n_groups=args.n_groups,
        n_variables=args.n_variables
    )
    
    # 分析结果
    fpr_data = analyze_results(results, type_counter, args.n_sim)
    
    # 保存 Markdown 报告
    if args.output:
        output_path = args.output
    else:
        output_path = str(Path(__file__).parent.parent / "references" / "arithmetic-detector-baseline-monte-carlo.md")
    
    save_baseline_report(fpr_data, args.n_sim, args.n_per_group, 
                        args.n_groups, args.n_variables, output_path)
    print(f"\n✅ 基线报告已保存: {output_path}")
    
    # JSON 输出（可被 skill 引用的结构化数据）
    if args.json:
        json_output = {
            "method": "monte_carlo_simulation",
            "date": "2026-06-06",
            "parameters": {
                "n_simulations": args.n_sim,
                "n_per_group": args.n_per_group,
                "n_groups": args.n_groups,
                "n_variables": args.n_variables,
                "total_datapoints": args.n_sim * args.n_groups * args.n_variables * args.n_per_group
            },
            "results": fpr_data,
            "all_zero_false_positive": all(info["count"] == 0 for info in fpr_data.values()),
            "false_positive_rate_upper_bound": f"< {1/args.n_sim:.4f}"
        }
        json_path = str(Path(__file__).parent.parent / "references" / "arithmetic-detector-baseline-monte-carlo.json")
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON基线数据已保存: {json_path}")


if __name__ == '__main__':
    main()
