#!/usr/bin/env python3
"""
aHash 图片快速初筛 — 来源: 公开方法/图片哈希检测
==========================================================
用 Average Hash + 6方向变体（原图/水平翻转/垂直翻转/90°/180°/270°）
全量交叉比对，标记候选重复对。

定位：Step⑥ 图片取证的第一道快速筛。
在 ELA/块重复/SIFT 之前跑。候选对交由后续精细化验证。

边界：
- 不替换 ELA/SIFT/块重复 — 仅做初筛
- 不检测局部裁剪复用 — aHash 对裁剪敏感
- 输出 JSON，供 unified_review 的 Step⑥ 接入

用法:
    python3 image_hash_screener.py <图片目录> [--threshold 6] [--format json|markdown]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageOps
except ImportError:
    print("❌ 需要 Pillow: pip install Pillow", file=sys.stderr)
    sys.exit(1)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def ahash(image: Image.Image, size: int = 8) -> str:
    """Average Hash: 缩放到 size×size，以均值为阈值二值化"""
    gray = image.convert("L").resize((size, size))
    pixels = list(gray.getdata())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if p >= avg else "0" for p in pixels)


def hamming(a: str, b: str) -> int:
    """汉明距离"""
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def variant_hashes(path: Path) -> Dict[str, str]:
    """对一张图生成6个方向的 aHash"""
    with Image.open(path) as img:
        base = img.convert("RGB")
        variants = {
            "original": base,
            "flip_lr": ImageOps.mirror(base),
            "flip_tb": ImageOps.flip(base),
            "rot90": base.rotate(90, expand=True),
            "rot180": base.rotate(180, expand=True),
            "rot270": base.rotate(270, expand=True),
        }
        return {name: ahash(variant) for name, variant in variants.items()}


def iter_images(paths: List[str]) -> List[Path]:
    """递归收集所有图片文件"""
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    files.append(child)
    return sorted(files)


def screen(paths: List[str], threshold: int = 6) -> dict:
    """
    主筛查逻辑
    返回: {tool, image_count, findings: [...], risk_level}
    """
    files = iter_images(paths)
    if len(files) < 2:
        return {
            "tool": "image_hash_screener",
            "input": paths,
            "image_count": len(files),
            "risk_level": "GREEN low",
            "findings": [],
            "evidence_files": [str(f) for f in files],
            "limitations": [
                "aHash 是粗粒度筛查；会漏掉局部裁剪复用，也会对视觉相似但内容不同的白底图产生假阳性。",
                "所有发现必须经 ELA/块重复/SIFT 二次验证。",
            ],
            "metadata": {"threshold": threshold},
        }

    # 预计算所有图片的6方向hash
    hash_map: Dict[str, Dict[str, str]] = {}
    for f in files:
        try:
            hash_map[str(f)] = variant_hashes(f)
        except Exception as e:
            print(f"⚠️ 跳过 {f.name}: {e}", file=sys.stderr)

    findings = []
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            left_path = str(files[i])
            right_path = str(files[j])
            if left_path not in hash_map or right_path not in hash_map:
                continue

            left_hashes = hash_map[left_path]
            right_hashes = hash_map[right_path]

            best: Tuple[int, str, str] = (999, "", "")
            for lv_name, lv_hash in left_hashes.items():
                for rv_name, rv_hash in right_hashes.items():
                    d = hamming(lv_hash, rv_hash)
                    if d < best[0]:
                        best = (d, lv_name, rv_name)

            if best[0] <= threshold:
                severity = "HIGH" if best[0] <= 3 else "MEDIUM" if best[0] <= 5 else "LOW"
                findings.append({
                    "method": "aHash_image_similarity",
                    "severity": severity,
                    "location": f"{files[i].name} vs {files[j].name}",
                    "description": (
                        f"图片哈希高度相似 (汉明距离={best[0]})，"
                        f"方向组合: {best[1]}↔{best[2]}。"
                        "可能是旋转/翻转复用，需 ELA/SIFT 二次验证。"
                    ),
                    "evidence_files": [str(files[i]), str(files[j])],
                    "statistics": {
                        "hamming_distance": best[0],
                        "left_variant": best[1],
                        "right_variant": best[2],
                        "left_size": files[i].stat().st_size,
                        "right_size": files[j].stat().st_size,
                    },
                    "recommendation": "用 ELA + SIFT 对该对做精细化验证。检查是否同一图片经旋转/翻转后用于不同实验。",
                })

    # 风险评级
    high_count = sum(1 for f in findings if f["severity"] == "HIGH")
    med_count = sum(1 for f in findings if f["severity"] == "MEDIUM")

    if high_count >= 3:
        risk = "RED severe"
    elif high_count >= 1 or med_count >= 5:
        risk = "ORANGE high"
    elif med_count >= 1:
        risk = "YELLOW moderate"
    else:
        risk = "GREEN low"

    return {
        "tool": "image_hash_screener",
        "input": paths,
        "image_count": len(files),
        "risk_level": risk,
        "findings": findings,
        "evidence_files": [str(f) for f in files],
        "limitations": [
            "aHash 是粗粒度筛查；会漏掉局部裁剪复用，也会对视觉相似但内容不同的白底图产生假阳性。",
            "所有 HIGH/MEDIUM 发现必须经 ELA/块重复/SIFT 二次验证，不可独立定罪。",
            "阈值 threshold=6 (默认) — 越小越严格，但假阴性率上升。",
        ],
        "metadata": {"threshold": threshold, "hash_size": 8, "variants": 6},
    }


def to_markdown(result: dict) -> str:
    lines = [
        "# aHash 图片快速初筛",
        "",
        f"- 图片数: {result['image_count']}",
        f"- 风险等级: {result['risk_level']}",
        f"- 阈值 (汉明距离): {result['metadata']['threshold']}",
        f"- 候选重复对: {len(result['findings'])}",
        "",
        "## 候选重复对",
        "",
    ]
    if not result["findings"]:
        lines.append("未发现候选重复。")
    else:
        lines.append("| # | 图片A | 图片B | 汉明距离 | 方向组合 | 严重性 |")
        lines.append("|---|---|---|---|---|---|")
        for idx, f in enumerate(result["findings"], 1):
            loc = f["location"]
            parts = loc.split(" vs ")
            a_name = parts[0] if len(parts) > 0 else "?"
            b_name = parts[1] if len(parts) > 1 else "?"
            lines.append(
                f"| {idx} | {a_name} | {b_name} | "
                f"{f['statistics']['hamming_distance']} | "
                f"{f['statistics']['left_variant']}↔{f['statistics']['right_variant']} | "
                f"{f['severity']} |"
            )
    lines.extend([
        "",
        "## ⚠️ 重要提示",
        "",
        "- 这是初筛结果，不是最终结论。",
        "- 所有候选对需经 ELA/块重复/SIFT 二次验证。",
        "- aHash 对白底为主的 PDF 整页提取图会产生自然高相似度（假阳性风险）。",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="aHash 图片快速初筛 — 检测学术论文图片旋转/翻转复用"
    )
    parser.add_argument("paths", nargs="+", help="图片文件或目录")
    parser.add_argument(
        "--threshold", "-t", type=int, default=6,
        help="汉明距离阈值 (默认: 6, 越小越严格)"
    )
    parser.add_argument(
        "--format", "-f", choices=["json", "markdown"], default="json"
    )
    parser.add_argument("--output", "-o", help="输出文件路径")
    args = parser.parse_args()

    result = screen(args.paths, args.threshold)

    if args.format == "markdown":
        output = to_markdown(result)
    else:
        output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"✅ 报告已保存: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
