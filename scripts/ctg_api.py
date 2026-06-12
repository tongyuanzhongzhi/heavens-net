#!/usr/bin/env python3
"""
天网·ClinicalTrials.gov 数据提取模块 v1.0

从 ClinicalTrials.gov API v2 提取结构化结果数据（outcome measures），
输出标准化的数值列表供检测器使用。

API 文档: https://clinicaltrials.gov/data-api/api

用法:
  from ctg_api import CTGClient
  client = CTGClient()
  
  # 搜索
  ncts = client.search_studies("diabetes", max_results=50)
  
  # 提取数值
  rows = client.get_study_results("NCT04292899")
  
  # 批量保存
  client.batch_extract(ncts, output_dir="./ctg_data/")
"""

import json
import time
from typing import Optional
from collections import Counter

try:
    import requests
except ImportError:
    raise ImportError("需要 requests 库: pip install requests")


API_BASE = "https://clinicaltrials.gov/api/v2"
USER_AGENT = "Heavens-Net/1.0 (academic fraud detection; mailto:keplerneo@proton.me)"


class CTGClient:
    """ClinicalTrials.gov API 客户端"""

    def __init__(self, rate_limit: float = 0.3, timeout: int = 30):
        """
        Args:
            rate_limit: API 请求间隔（秒），建议 ≥0.2
            timeout: 单次请求超时（秒）
        """
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': USER_AGENT,
            'Accept': 'application/json'
        })

    # ── 搜索 ──────────────────────────────────────────────

    def search_studies(self, condition: str,
                       page_size: int = 100,
                       max_pages: int = 5,
                       has_results: bool = True) -> list[str]:
        """
        搜索试验

        Args:
            condition: 疾病/关键词（如 "diabetes", "breast cancer"）
            page_size: 每页数量
            max_pages: 最大翻页数
            has_results: 是否只返回有结果数据的试验

        Returns:
            NCT ID 列表
        """
        all_nct = []
        page_token = None

        for i in range(max_pages):
            params = {
                "query.cond": condition,
                "pageSize": page_size,
                "format": "json"
            }
            if has_results:
                params["query.term"] = "AREA[HasResults]true"
            if page_token:
                params["pageToken"] = page_token

            try:
                r = self.session.get(
                    f"{API_BASE}/studies",
                    params=params,
                    timeout=self.timeout
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                print(f"  [搜索] 第 {i+1} 页失败: {e}")
                break

            studies = data.get("studies", [])
            for s in studies:
                nct = s.get("protocolSection", {}) \
                         .get("identificationModule", {}) \
                         .get("nctId")
                if nct:
                    all_nct.append(nct)

            page_token = data.get("nextPageToken", "")
            if not page_token or not studies:
                break

            time.sleep(self.rate_limit)

        return all_nct

    def search_multi_conditions(self, conditions: list[str],
                                per_condition: int = 50) -> list[str]:
        """搜索多个疾病，返回去重后的 NCT ID 列表"""
        all_nct = []
        for cond in conditions:
            ncts = self.search_studies(cond, page_size=per_condition, max_pages=2)
            all_nct.extend(ncts)
            time.sleep(0.5)
        return list(dict.fromkeys(all_nct))

    # ── 详细结果提取 ──────────────────────────────────────

    def get_study_results(self, nct_id: str) -> list[dict]:
        """
        提取单个试验的结构化数值数据

        数据来源（按优先级）:
          1. resultsSection.outcomeMeasuresModule.outcomeMeasures
             → classes → categories → measurements
             → analyses（p值、效应量）
             → denoms（分母计数）

        Returns:
            [{nct, outcome, type, group, category, value, spread, units, source}, ...]
        """
        url = f"{API_BASE}/studies/{nct_id}?format=json"
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"  [{nct_id}] 请求失败: {e}")
            return []

        rows = []
        ome = data.get('resultsSection', {}).get('outcomeMeasuresModule', {})
        oms = ome.get('outcomeMeasures', [])

        for om in oms:
            title = om.get('title', '')[:120]
            otype = om.get('type', '')       # PRIMARY / SECONDARY
            unit = om.get('unitOfMeasure', '')
            param_type = om.get('paramType', '')

            # 路径 A: classes → categories → measurements
            for cls in om.get('classes', []):
                cls_title = cls.get('title', '')[:50]
                for cat in cls.get('categories', []):
                    cat_title = cat.get('title', '')[:50]
                    for m in cat.get('measurements', []):
                        val = self._safe_float(m.get('value'))
                        rows.append({
                            'nct': nct_id,
                            'outcome': title,
                            'type': otype,
                            'param_type': param_type,
                            'group': cls_title,
                            'category': cat_title,
                            'value': val,
                            'spread': m.get('spread', ''),
                            'units': unit or m.get('units', ''),
                            'source': 'class_category'
                        })

            # 路径 B: analyses（p值、效应量、置信区间）
            for an in om.get('analyses', []):
                param = an.get('paramType', '')
                grp_ids = an.get('groupIds', [])
                grp_label = ','.join(grp_ids) if grp_ids else 'analysis'

                fields = {
                    'paramVal': an.get('paramValue'),
                    'pVal': an.get('pValue'),
                    'ciLow': an.get('ciLowerLimit'),
                    'ciUp': an.get('ciUpperLimit')
                }
                for lbl, raw_val in fields.items():
                    val = self._safe_float(raw_val)
                    if val is not None:
                        rows.append({
                            'nct': nct_id,
                            'outcome': title,
                            'type': otype,
                            'param_type': param,
                            'group': grp_label,
                            'category': lbl,
                            'value': val,
                            'spread': '',
                            'units': unit,
                            'source': f'analysis_{lbl}'
                        })

            # 路径 C: denoms（分母计数）
            for dn in om.get('denoms', []):
                for ct in dn.get('counts', []):
                    val = self._safe_float(ct.get('value'))
                    if val is not None:
                        rows.append({
                            'nct': nct_id,
                            'outcome': title,
                            'type': otype,
                            'param_type': 'DENOM',
                            'group': ct.get('groupId', '?'),
                            'category': 'denominator',
                            'value': val,
                            'spread': '',
                            'units': ct.get('units', ''),
                            'source': 'denom'
                        })

        return rows

    # ── 批量操作 ──────────────────────────────────────────

    def batch_extract(self, nct_ids: list[str],
                      output_dir: str = "./ctg_data/",
                      max_trials: int = 200) -> dict:
        """
        批量提取多个试验的数据

        Args:
            nct_ids: NCT ID 列表
            output_dir: 输出目录
            max_trials: 最多处理的试验数

        Returns:
            {
                "downloaded": int,
                "total_rows": int,
                "errors": int,
                "output_files": [str]
            }
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        all_rows = []
        downloaded = 0
        errors = 0

        for nct in nct_ids[:max_trials]:
            try:
                rows = self.get_study_results(nct)
                if rows:
                    all_rows.extend(rows)
                    downloaded += 1
                    # 每10个试验保存一次中间结果
                    if downloaded % 50 == 0:
                        _save_rows(all_rows, os.path.join(output_dir, '_checkpoint.json'))
            except Exception as e:
                errors += 1
            time.sleep(self.rate_limit)

        # 保存最终结果
        output_path = os.path.join(output_dir, 'ctg_extracted.json')
        _save_rows(all_rows, output_path)

        return {
            "downloaded": downloaded,
            "total_rows": len(all_rows),
            "errors": errors,
            "output_file": output_path
        }

    def get_trial_metadata(self, nct_id: str) -> dict:
        """提取试验的基本元数据（标题、状态、阶段等）"""
        url = f"{API_BASE}/studies/{nct_id}?format=json"
        try:
            r = self.session.get(url, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return {}

        ps = data.get('protocolSection', {})
        im = ps.get('identificationModule', {})
        sm = ps.get('statusModule', {})
        dm = ps.get('designModule', {})

        return {
            'nct_id': nct_id,
            'title': im.get('briefTitle', '')[:200],
            'official_title': im.get('officialTitle', '')[:200],
            'status': sm.get('overallStatus', ''),
            'phase': dm.get('phases', []),
            'enrollment': dm.get('enrollmentInfo', {}).get('count', None),
            'study_type': dm.get('studyType', ''),
            'has_results': bool(sm.get('resultsFirstPostDate')),
            'url': f'https://clinicaltrials.gov/study/{nct_id}'
        }

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        """安全转换为浮点数"""
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def stats(rows: list[dict]) -> dict:
        """统计提取数据的分布"""
        if not rows:
            return {}

        ncts = set(r['nct'] for r in rows)
        sources = Counter(r['source'] for r in rows)
        types = Counter(r['type'] for r in rows)
        value_count = sum(1 for r in rows if r['value'] is not None)

        return {
            'total_rows': len(rows),
            'unique_trials': len(ncts),
            'value_rows': value_count,
            'by_source': dict(sources),
            'by_type': dict(types)
        }


def _save_rows(rows: list[dict], path: str):
    """保存行数据到 JSON"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False, default=str)


# ── CLI ──────────────────────────────────────────────────

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='天网·ClinicalTrials.gov 数据提取工具 v1.0'
    )
    parser.add_argument('action', choices=['search', 'extract', 'meta'],
                        help='search: 搜索试验 / extract: 提取数值 / meta: 查看元数据')
    parser.add_argument('--condition', '-c', default='diabetes',
                        help='搜索关键词 (默认: diabetes)')
    parser.add_argument('--nct', '-n', help='NCT ID（如 NCT04292899）')
    parser.add_argument('--nct-file', help='NCT ID 列表文件（每行一个）')
    parser.add_argument('--output', '-o', help='输出文件/目录')
    parser.add_argument('--max-trials', type=int, default=100,
                        help='最多处理试验数 (默认: 100)')
    parser.add_argument('--multi', nargs='+',
                        help='搜索多个疾病，空格分隔')

    args = parser.parse_args()
    client = CTGClient()

    if args.action == 'search':
        if args.multi:
            ncts = client.search_multi_conditions(args.multi)
        else:
            ncts = client.search_studies(args.condition)
        print(f'找到 {len(ncts)} 个试验:')
        for n in ncts[:20]:
            print(f'  {n}')
        if len(ncts) > 20:
            print(f'  ... 还有 {len(ncts) - 20} 个')
        if args.output:
            with open(args.output, 'w') as f:
                f.write('\n'.join(ncts))

    elif args.action == 'extract':
        if args.nct:
            nct_ids = [args.nct]
        elif args.nct_file:
            with open(args.nct_file) as f:
                nct_ids = [l.strip() for l in f if l.strip()]
        else:
            print("请指定 --nct 或 --nct-file")
            return

        output_dir = args.output or './ctg_data/'
        result = client.batch_extract(nct_ids, output_dir=output_dir,
                                      max_trials=args.max_trials)
        print(f"\n提取完成:")
        print(f"  试验数: {result['downloaded']}")
        print(f"  数据行: {result['total_rows']}")
        print(f"  错误数: {result['errors']}")
        print(f"  输出: {result['output_file']}")

    elif args.action == 'meta':
        if not args.nct:
            print("请指定 --nct")
            return
        meta = client.get_trial_metadata(args.nct)
        for k, v in meta.items():
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
