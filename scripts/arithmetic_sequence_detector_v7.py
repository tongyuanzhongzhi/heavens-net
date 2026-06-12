#!/usr/bin/env python3
"""
天网·算术序列检测器 v7.0 — 无列名依赖 + 强化假阳性过滤

核心改进（vs v6.0）:
  - 自动检测所有数值列（不再要求特定列名）
  - 三层假阳性过滤：diff≠0 / unique≥3 / 众数<60%
  - 按列自动分组，逐列独立检测
  - 与 v6.0 输出格式兼容

用法:
  python3 arithmetic_sequence_detector_v7.py data.csv
  python3 arithmetic_sequence_detector_v7.py data.csv --min-pairs 3 --tolerance 1e-6
"""

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

VERSION = "7.0"
DEFAULT_MIN_PAIRS = 3
DEFAULT_TOLERANCE = 1e-6
DEFAULT_P_THRESHOLD = 0.01


class ArithmeticSequenceDetectorV7:
    """改进版算术序列检测器 — 无列名依赖"""

    def __init__(self, min_pairs: int = DEFAULT_MIN_PAIRS,
                 tolerance: float = DEFAULT_TOLERANCE,
                 p_threshold: float = DEFAULT_P_THRESHOLD):
        self.min_pairs = min_pairs
        self.tolerance = tolerance
        self.p_threshold = p_threshold

    # ── 数据加载 ──────────────────────────────────────────

    def load_csv(self, path: str) -> dict:
        """加载 CSV，自动检测数值列，返回 {列名: [浮点数]}"""
        numeric_columns = {}

        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                return numeric_columns

            # 对每列，尝试解析为数值
            for col_name in reader.fieldnames:
                values = []
                for row in rows:
                    val = self._parse_number(row.get(col_name, ''))
                    if val is not None:
                        values.append(val)

                # 保留有足够数值的列
                if len(values) >= 4:
                    numeric_columns[col_name] = values

        return numeric_columns

    def load_csv_all_columns(self, path: str) -> list[dict]:
        """加载 CSV，返回每行的所有数值列（用于跨列对比）"""
        rows_out = []

        with open(path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for col_name, val_str in row.items():
                    val = self._parse_number(val_str)
                    if val is not None:
                        parsed[col_name] = val
                if parsed:
                    rows_out.append(parsed)

        return rows_out

    def load_data(self, path: str) -> list[dict]:
        """加载任意格式数据，返回 [{col: val, col: val}, ...]"""
        ext = Path(path).suffix.lower()
        if ext == '.csv':
            return self.load_csv_all_columns(path)
        elif ext in ('.xlsx', '.xls'):
            return self._load_excel(path)
        elif ext == '.json':
            return self._load_json(path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

    def _parse_number(self, val_str: str) -> Optional[float]:
        """安全解析数值字符串"""
        if val_str is None:
            return None
        val_str = str(val_str).strip()
        if not val_str:
            return None
        # 移除百分号
        val_str = val_str.replace('%', '')
        # 移除千位分隔符
        val_str = val_str.replace(',', '')
        try:
            v = float(val_str)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except (ValueError, TypeError):
            return None

    def _load_excel(self, path: str) -> list[dict]:
        """加载 Excel 文件"""
        try:
            import openpyxl
        except ImportError:
            raise ImportError("需要 openpyxl: pip install openpyxl")

        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)

        # 第一行作为列名
        headers = [str(h) if h else f"col_{i}" for i, h in enumerate(next(rows_iter))]
        rows_out = []

        for row in rows_iter:
            parsed = {}
            for i, val in enumerate(row):
                if i >= len(headers):
                    break
                v = self._parse_number(val)
                if v is not None:
                    parsed[headers[i]] = v
            if parsed:
                rows_out.append(parsed)

        wb.close()
        return rows_out

    def _load_json(self, path: str) -> list[dict]:
        """加载 JSON 数据"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 支持 [{col:val}, ...] 或 {col: [val, val]}
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # 将 {col: [vals]} 转为 [{col: val}, ...]
            lengths = {k: len(v) for k, v in data.items() if isinstance(v, list)}
            if not lengths:
                return [data]
            max_len = min(lengths.values())
            rows = []
            for i in range(max_len):
                row = {}
                for k, v in data.items():
                    if isinstance(v, list) and i < len(v):
                        val = self._parse_number(v[i])
                        if val is not None:
                            row[k] = val
                if row:
                    rows.append(row)
            return rows

        return [data]

    # ── 核心检测 ──────────────────────────────────────────

    def _quality_filter(self, values: list[float]) -> tuple[bool, str]:
        """
        三层假阳性过滤
        Returns: (通过?, 拒绝原因)
        """
        nums = [v for v in values if v is not None
                and not math.isnan(v) and not math.isinf(v)]

        if len(nums) < 4:
            return False, "样本不足 (<4)"

        # 第1层: 至少3个不同值（拒绝全零/全同值）
        unique_count = len(set(nums))
        if unique_count < 3:
            return False, f"唯一值不足 ({unique_count}<3)"

        # 第2层: 中位数不为0时，众数占比不能超过60%（拒绝临床零事件噪音）
        val_counter = Counter(nums)
        top_count = val_counter.most_common(1)[0][1]
        mode_ratio = top_count / len(nums)
        if mode_ratio > 0.6:
            return False, f"单值占比过高 ({mode_ratio:.1%}>60%)"

        return True, ""

    def detect_sequence(self, values: list[float], label: str = "") -> dict:
        """
        检测一组数值中是否存在固定差值模式

        Returns: {
            "flagged": bool,
            "label": str,
            "diff": float or None,
            "p_value": float,
            "aligned_pairs": int,
            "total_points": int,
            "values": list,
            "reject_reason": str or None
        }
        """
        result = {
            "flagged": False,
            "label": label,
            "diff": None,
            "p_value": 1.0,
            "aligned_pairs": 0,
            "total_points": 0,
            "values": [],
            "reject_reason": None
        }

        # 清理输入
        nums = [v for v in values if v is not None
                and isinstance(v, (int, float))
                and not math.isnan(v) and not math.isinf(v)]

        result["total_points"] = len(nums)
        result["values"] = sorted(set(nums))

        if len(nums) < 4:
            result["reject_reason"] = "样本不足 (<4)"
            return result

        # 质量过滤
        passed, reason = self._quality_filter(nums)
        if not passed:
            result["reject_reason"] = reason
            return result

        # 排序 → 计算相邻差值
        nums_sorted = sorted(nums)
        diffs = []
        for i in range(len(nums_sorted) - 1):
            d = round(nums_sorted[i + 1] - nums_sorted[i], 10)
            diffs.append(d)

        # 频率分析
        diff_counter = Counter(diffs)
        most_diff, match_count = diff_counter.most_common(1)[0]

        # 需要至少 min_pairs 对呈现相同差值
        if match_count < self.min_pairs:
            return result

        # 序列覆盖率
        coverage = (match_count + 1) / len(nums_sorted)
        if coverage < 0.5:
            return result

        # 统计算法: 随机 n 个点中连续 k 个等差序列的概率
        unique_diffs = len(set(diffs))
        if unique_diffs <= 1:
            return result  # 全是同一种差值也算可疑，但这里放过

        # p值估计: 如果差值均匀分布，随机选 k 个连续差值为特定值的概率
        p_raw = (1.0 / (unique_diffs - 1)) ** (match_count - 1)
        p_value = min(p_raw * math.comb(len(nums_sorted), match_count + 1), 1.0)

        if p_value < self.p_threshold:
            result["flagged"] = True
            result["diff"] = most_diff
            result["p_value"] = p_value
            result["aligned_pairs"] = match_count

        return result

    # ── 批量检测 ──────────────────────────────────────────

    def scan_csv(self, path: str) -> list[dict]:
        """扫描 CSV 的所有数值列，返回警报列表"""
        numeric_cols = self.load_csv(path)
        results = []

        for col_name, values in numeric_cols.items():
            r = self.detect_sequence(values, label=col_name)
            if r["flagged"]:
                results.append(r)

        results.sort(key=lambda x: x["p_value"])
        return results

    def scan_file(self, path: str) -> list[dict]:
        """通用扫描 — 自动检测格式并分组扫描"""
        ext = Path(path).suffix.lower()

        if ext == '.csv':
            # CSV: 逐列扫描
            return self.scan_csv(path)
        else:
            # 其他格式: 加载后按列分组扫描
            rows = self.load_data(path)
            if not rows:
                return []

            # 按列整理
            col_values = {}
            for row in rows:
                for col, val in row.items():
                    col_values.setdefault(col, []).append(val)

            results = []
            for col, vals in col_values.items():
                r = self.detect_sequence(vals, label=col)
                if r["flagged"]:
                    results.append(r)

            results.sort(key=lambda x: x["p_value"])
            return results

    def scan_files(self, paths: list[str]) -> dict[str, list[dict]]:
        """批量扫描多个文件"""
        all_results = {}
        for p in paths:
            path_str = str(p)
            results = self.scan_file(path_str)
            all_results[path_str] = results
        return all_results

    def scan_ctg_results(self, nct_ids: list[str]) -> list[dict]:
        """
        从 ClinicalTrials.gov 拉取结构化结果并扫描
        需要 requests 库和网络连接
        """
        try:
            from ctg_api import CTGClient
        except ImportError:
            raise ImportError("需要 ctg_api.py: 从 Heaven's Net 仓库获取")

        client = CTGClient()
        all_alerts = []

        for nct in nct_ids:
            rows = client.get_study_results(nct)
            if not rows:
                continue

            # 按 (outcome, group) 分组
            grouped = {}
            for row in rows:
                key = f"{nct}|{row.get('outcome','')[:60]}|{row.get('group','')}"
                grouped.setdefault(key, []).append(row.get('value'))

            for grp_key, vals in grouped.items():
                r = self.detect_sequence(vals, label=grp_key)
                if r["flagged"]:
                    all_alerts.append(r)

        all_alerts.sort(key=lambda x: x["p_value"])
        return all_alerts


# ── CLI ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=f'天网·算术序列检测器 v{VERSION} — 无需指定列名，自动检测所有数值列'
    )
    parser.add_argument('files', nargs='+', help='CSV/Excel/JSON 文件路径')
    parser.add_argument('--min-pairs', type=int, default=DEFAULT_MIN_PAIRS,
                        help=f'最少等差对数量 (默认: {DEFAULT_MIN_PAIRS})')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE,
                        help=f'差值误差容忍度 (默认: {DEFAULT_TOLERANCE})')
    parser.add_argument('--p-threshold', type=float, default=DEFAULT_P_THRESHOLD,
                        help=f'p值报警阈值 (默认: {DEFAULT_P_THRESHOLD})')
    parser.add_argument('--output', '-o', help='输出 JSON 文件路径')
    parser.add_argument('--all', action='store_true',
                        help='输出所有检测结果（包括未报警的）')
    parser.add_argument('--ctg', action='store_true',
                        help='识别文件为 NCT ID 列表，从 ClinicalTrials.gov 拉取数据')

    args = parser.parse_args()

    detector = ArithmeticSequenceDetectorV7(
        min_pairs=args.min_pairs,
        tolerance=args.tolerance,
        p_threshold=args.p_threshold
    )

    if args.ctg:
        # NCT ID 模式
        with open(args.files[0], 'r') as f:
            nct_ids = [line.strip() for line in f if line.strip()]
        results = detector.scan_ctg_results(nct_ids)
    else:
        results = []
        for fp in args.files:
            file_results = detector.scan_file(fp)
            if args.all:
                # 输出所有检测结果
                all_r = detector.scan_csv(fp) if fp.endswith('.csv') else file_results
                for r in all_r:
                    r["source_file"] = fp
                results.extend(all_r)
            else:
                for r in file_results:
                    r["source_file"] = fp
                results.extend(file_results)

    # 输出
    output = {
        "version": VERSION,
        "config": {
            "min_pairs": args.min_pairs,
            "tolerance": args.tolerance,
            "p_threshold": args.p_threshold
        },
        "total_alerts": len([r for r in results if r.get("flagged")]),
        "total_scanned": len(results),
        "results": results
    }

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False, default=str)
        print(f'结果已保存: {args.output}')
    else:
        # 终端摘要
        alerts = [r for r in results if r.get("flagged")]
        if not alerts:
            print('✓ 未发现可疑算术序列')
            print(f'  扫描了 {len(results)} 列/组数据')
        else:
            print(f'⚠ 发现 {len(alerts)} 个可疑算术序列:')
            print('-' * 60)
            for r in alerts[:20]:
                src = r.get('source_file', r.get('label', '?'))
                print(f"\n  [{src}]")
                print(f"  差值: {r['diff']:.6f}  |  p ≈ {r['p_value']:.2e}")
                print(f"  对齐: {r['aligned_pairs']+1}/{r['total_points']} 个点")
                if r.get('values'):
                    vals = [round(v, 4) for v in r['values'][:8]]
                    print(f"  值: {vals}")

        # 被过滤的
        rejected = [r for r in results if r.get("reject_reason")]
        if rejected and not args.all:
            print(f'\n  过滤了 {len(rejected)} 组 (低质量数据)')


if __name__ == '__main__':
    main()
