#!/usr/bin/env python3
"""
Western Blot / 凝胶电泳泳道拼接检测 — 来源: 公开方法/泳道审计
========================================================================
两阶段检测：
1. 列强度剖面 → z-score 检测垂直分界线（拼接痕迹）
2. 泳道哈希 → 汉明距离找重复泳道

定位：Step⑥ 图片取证 WB 专项。
与现有 analyze_wb_bands.py 互补：
- analyze_wb_bands.py → 条带级互相关（band-level）
- 本脚本 → 泳道级拼接+重复（lane-level）
两者不冲突，可同时运行。

用法:
    python3 blot_gel_lane_audit.py <图片路径...> [--lanes 8] [--seam-z 3.0]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    print("❌ 需要 Pillow: pip install Pillow", file=sys.stderr)
    sys.exit(1)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def hamming(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))


def column_profile(image: Image.Image) -> List[float]:
    """计算每列的平均灰度，返回长度为 width 的列表"""
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.load()
    return [sum(pixels[x, y] for y in range(height)) / height for x in range(width)]


def detect_boundaries(profile: List[float], z_threshold: float) -> List[int]:
    """
    z-score 检测垂直分界线
    对相邻列的强度差分做 z-score，标记 ≥ z_threshold 的列
    """
    if len(profile) < 3:
        return []
    diffs = [abs(profile[i + 1] - profile[i]) for i in range(len(profile) - 1)]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / max(len(diffs) - 1, 1)
    sd = variance ** 0.5
    if sd == 0:
        return []
    return [idx for idx, d in enumerate(diffs) if (d - mean_diff) / sd >= z_threshold]


def lane_hashes(image: Image.Image, lanes: int) -> List[str]:
    """
    将图片等分为 lanes 个泳道，每个泳道做 8×16 aHash
    返回 lanes 个哈希字符串
    """
    gray = image.convert("L")
    width, height = gray.size
    lane_width = max(width // lanes, 1)
    hashes = []
    for lane in range(lanes):
        left = lane * lane_width
        right = width if lane == lanes - 1 else min((lane + 1) * lane_width, width)
        crop = gray.crop((left, 0, right, height)).resize((8, 16))
        pixels = list(crop.getdata())
        avg = sum(pixels) / len(pixels)
        hashes.append("".join("1" if p >= avg else "0" for p in pixels))
    return hashes


def audit(paths: List[str], lanes: int = 8, seam_z: float = 3.0,
          lane_threshold: int = 6) -> dict:
    """
    主检测逻辑
    """
    # 收集图片
    files: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    files.append(child)

    findings = []
    for fp in files:
        try:
            with Image.open(fp) as img:
                # 阶段1: 分界线检测
                profile = column_profile(img)
                boundaries = detect_boundaries(profile, seam_z)
                if len(boundaries) >= 2:
                    severity = "HIGH" if len(boundaries) >= 8 else "MEDIUM"
                    findings.append({
                        "method": "blot_gel_boundary_screen",
                        "severity": severity,
                        "location": fp.name,
                        "description": (
                            f"检测到 {len(boundaries)} 处垂直强度突变"
                            f"（z≥{seam_z}），可能为泳道拼接痕迹。"
                        ),
                        "evidence_files": [str(fp)],
                        "statistics": {
                            "boundary_count": len(boundaries),
                            "boundary_positions": boundaries[:25],
                            "image_size": list(img.size),
                        },
                        "recommendation": "检查未裁剪原始图片，确认泳道是否被拼接或重排。",
                    })

                # 阶段2: 重复泳道检测
                h_list = lane_hashes(img, lanes)
                for i in range(len(h_list)):
                    for j in range(i + 1, len(h_list)):
                        d = hamming(h_list[i], h_list[j])
                        if d <= lane_threshold:
                            sev = "HIGH" if d <= 3 else "MEDIUM"
                            findings.append({
                                "method": "blot_gel_repeated_lane",
                                "severity": sev,
                                "location": f"{fp.name}: lane {i+1} vs lane {j+1}",
                                "description": (
                                    f"泳道 {i+1} 与泳道 {j+1} 强度模式高度相似"
                                    f"（汉明距离={d}），可能为重复泳道。"
                                ),
                                "evidence_files": [str(fp)],
                                "statistics": {
                                    "hamming_distance": d,
                                    "lanes": [i + 1, j + 1],
                                    "estimated_total_lanes": lanes,
                                },
                                "recommendation": "检查两个泳道是否为重复、相邻技术重复、或有效的重复对照。",
                            })
        except Exception as e:
            print(f"⚠️ 跳过 {fp.name}: {e}", file=sys.stderr)

    # 风险评级
    high = sum(1 for f in findings if f["severity"] == "HIGH")
    med = sum(1 for f in findings if f["severity"] == "MEDIUM")
    if high >= 3:
        risk = "RED severe"
    elif high >= 1 or med >= 5:
        risk = "ORANGE high"
    elif med >= 1:
        risk = "YELLOW moderate"
    else:
        risk = "GREEN low"

    return {
        "tool": "blot_gel_lane_audit",
        "input": paths,
        "image_count": len(files),
        "risk_level": risk,
        "findings": findings,
        "evidence_files": [str(f) for f in files],
        "limitations": [
            "泳道分割是近似估计；以未裁剪原始扫描图为最终判断依据。",
            "z-score 阈值检测对窄泳道图片敏感度下降。",
            "重复泳道检测基于 8×16 aHash——会漏掉小幅修改的复用。",
        ],
        "metadata": {
            "estimated_lanes": lanes,
            "seam_z_threshold": seam_z,
            "lane_hamming_threshold": lane_threshold,
        },
    }


def to_markdown(result: dict) -> str:
    lines = [
        "# Western Blot / 凝胶泳道检测",
        "",
        f"- 图片数: {result['image_count']}",
        f"- 风险等级: {result['risk_level']}",
        f"- 泳道数(估计): {result['metadata']['estimated_lanes']}",
        f"- 分界z阈值: {result['metadata']['seam_z_threshold']}",
        f"- 发现: {len(result['findings'])}",
        "",
        "## 发现",
        "",
    ]
    if not result["findings"]:
        lines.append("未发现泳道异常。")
    else:
        boundary_findings = [f for f in result["findings"] if "boundary" in f["method"]]
        lane_findings = [f for f in result["findings"] if "repeated_lane" in f["method"]]

        if boundary_findings:
            lines.append("### 分界线（拼接痕迹）")
            lines.append("")
            for f in boundary_findings:
                lines.append(f"- **{f['location']}**: {f['description']}")
            lines.append("")

        if lane_findings:
            lines.append("### 重复泳道")
            lines.append("")
            lines.append("| 图片 | 泳道对 | 汉明距离 | 严重性 |")
            lines.append("|---|---|---|---|")
            for f in lane_findings:
                parts = f["location"].split(": ")
                img_name = parts[0]
                lane_pair = parts[1] if len(parts) > 1 else "?"
                lines.append(
                    f"| {img_name} | {lane_pair} | "
                    f"{f['statistics']['hamming_distance']} | {f['severity']} |"
                )

    lines.extend([
        "",
        "## 局限性",
        "",
        "- 泳道分割是近似估计。",
        "- 需以未裁剪原始扫描图做最终判断。",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="WB/凝胶泳道拼接与重复检测"
    )
    parser.add_argument("paths", nargs="+", help="WB/凝胶图片文件或目录")
    parser.add_argument("--lanes", type=int, default=8,
                       help="估计泳道数 (默认: 8)")
    parser.add_argument("--seam-z", type=float, default=3.0,
                       help="分界线z-score阈值 (默认: 3.0)")
    parser.add_argument("--lane-threshold", type=int, default=6,
                       help="泳道重复汉明距离阈值 (默认: 6)")
    parser.add_argument("--format", "-f", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", "-o", help="输出文件路径")
    args = parser.parse_args()

    result = audit(args.paths, args.lanes, args.seam_z, args.lane_threshold)

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
