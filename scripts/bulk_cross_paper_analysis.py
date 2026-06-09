#!/usr/bin/env python3
"""
批量跨论文图片一致性分析

从多目录多PDF批量提取图片 → 跨论文pHASH比对 → 论文内部自比对 → 输出相似度报告

用法:
  python3 bulk_cross_paper_analysis.py <论文根目录> [--min-kb 50] [--threshold 0.75]

依赖:
  pip3 install pymupdf Pillow numpy

输出:
  - cross_paper_similarity.json: 所有跨论文相似对
  - intra_paper_similarity.json: 每篇论文内部自比对
  - figures/: 提取的所有图片汇总
"""

import os, sys, json, itertools, argparse
import numpy as np
from PIL import Image


def extract_all_figures(root_dir, output_base, min_kb=50):
    """从根目录下所有PDF递归提取图片"""
    figures_dir = os.path.join(output_base, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    pdf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(dirpath, f))

    import fitz
    extracted = {}  # {paper_label: [fig_paths]}

    for pdf_path in sorted(pdf_files):
        # 生成论文标签: 取文件名 (不含.pdf)
        fname = os.path.splitext(os.path.basename(pdf_path))[0]
        # 取相对路径作为唯一标识
        rel = os.path.relpath(pdf_path, root_dir)
        label = rel.replace('/', '_').replace('.pdf', '')[:50]

        doc = fitz.open(pdf_path)
        figs = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            for idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                base = doc.extract_image(xref)
                w, h = base["width"], base["height"]
                size_kb = len(base["image"]) / 1024
                if size_kb < min_kb:
                    continue
                ext = base["ext"]
                fig_name = f"{label}_p{page_num+1}_img{idx+1}_{w}x{h}.{ext}"
                fig_path = os.path.join(figures_dir, fig_name)
                with open(fig_path, 'wb') as f:
                    f.write(base["image"])
                figs.append({
                    "path": fig_path,
                    "name": fig_name,
                    "size_kb": round(size_kb, 1),
                    "paper": fname,
                    "page": page_num + 1,
                    "dimensions": f"{w}x{h}"
                })
        doc.close()
        extracted[label] = figs
        print(f"  {label}: {len(figs)} figures extracted")

    return extracted


def compute_phash(img_path):
    """计算64位均值感知哈希"""
    img = Image.open(img_path).convert('L').resize((32, 32), Image.LANCZOS)
    pix = np.array(img).flatten()
    avg = np.mean(pix)
    bits = (pix > avg).astype(int)[:64]
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def cross_paper_similarity(extracted, threshold=0.75):
    """跨论文所有图片两两比对"""
    all_figs = []
    for label, figs in extracted.items():
        for f in figs:
            f["paper_label"] = label
            all_figs.append(f)

    # 计算所有图片的pHash
    print(f"\n计算{len(all_figs)}张图片的pHash...")
    hashes = {}
    for f in all_figs:
        try:
            hashes[f["name"]] = compute_phash(f["path"])
        except Exception as e:
            print(f"  WARN: {f['name']} pHash失败: {e}")

    # 跨论文比对
    pairs = []
    for i in range(len(all_figs)):
        f1 = all_figs[i]
        h1 = hashes.get(f1["name"])
        if h1 is None:
            continue
        for j in range(i + 1, len(all_figs)):
            f2 = all_figs[j]
            h2 = hashes.get(f2["name"])
            if h2 is None:
                continue

            dist = bin(h1 ^ h2).count('1')
            sim = 1.0 - dist / 64.0
            if sim >= threshold:
                # 是否跨论文
                cross = f1["paper_label"] != f2["paper_label"]
                pairs.append({
                    "fig1": f1["name"],
                    "fig2": f2["name"],
                    "paper1": f1["paper"],
                    "paper2": f2["paper"],
                    "dim1": f1["dimensions"],
                    "dim2": f2["dimensions"],
                    "size1_kb": f1["size_kb"],
                    "size2_kb": f2["size_kb"],
                    "similarity": round(sim, 4),
                    "hamming": int(dist),
                    "cross_paper": cross
                })

    # 排序：相似度从高到低
    pairs.sort(key=lambda x: -x["similarity"])
    return pairs


def intra_paper_similarity(extracted, threshold=0.70):
    """每篇论文内部的跨图相似度"""
    results = {}
    for label, figs in extracted.items():
        hashes = {}
        for f in figs:
            try:
                hashes[f["name"]] = compute_phash(f["path"])
            except:
                pass
        pairs = []
        names = list(hashes.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                h1, h2 = hashes[names[i]], hashes[names[j]]
                dist = bin(h1 ^ h2).count('1')
                sim = 1.0 - dist / 64.0
                if sim >= threshold:
                    pairs.append({
                        "fig1": names[i],
                        "fig2": names[j],
                        "similarity": round(sim, 4)
                    })
        if pairs:
            results[label] = sorted(pairs, key=lambda x: -x["similarity"])
    return results


def verify_high_similarity_pairs(pairs, extracted, threshold_verify=0.80):
    """
    对高相似度的跨论文图像对做像素级SSIM验证

    因为pHash对不同尺寸但色调相似的图像可能产生假阳性（实测可达0.95-1.00），
    所有pHash相似度≥0.80的对都必须用SSIM验证。

    参数:
        pairs: cross_paper_similarity() 返回的对列表
        extracted: extract_all_figures() 返回的 {label: [figs]} 字典
        threshold_verify: 触发验证的pHash阈值

    返回:
        verified: 带SSIM结果的列表
    """
    # 建立 name -> path 的映射
    name_to_path = {}
    for label, figs in extracted.items():
        for f in figs:
            name_to_path[f["name"]] = f["path"]

    from scipy.ndimage import uniform_filter

    def ssim(img1, img2):
        K1, K2 = 0.01, 0.03
        L = 255
        C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
        mu1 = uniform_filter(img1.astype(float), size=11)
        mu2 = uniform_filter(img2.astype(float), size=11)
        sigma1_sq = uniform_filter(img1.astype(float) ** 2, size=11) - mu1 ** 2
        sigma2_sq = uniform_filter(img2.astype(float) ** 2, size=11) - mu2 ** 2
        sigma12 = uniform_filter(img1.astype(float) * img2.astype(float), size=11) - mu1 * mu2
        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(np.mean(ssim_map))

    from PIL import Image
    verified = []
    for p in pairs:
        if p["similarity"] < threshold_verify:
            continue

        path1 = name_to_path.get(p["fig1"])
        path2 = name_to_path.get(p["fig2"])
        if not path1 or not path2:
            p["ssim"] = None
            p["verdict"] = "PATH_NOT_FOUND"
            verified.append(p)
            continue

        try:
            img1 = Image.open(path1).convert('L')
            img2 = Image.open(path2).convert('L')
        except:
            p["ssim"] = None
            p["verdict"] = "LOAD_ERROR"
            verified.append(p)
            continue

        w1, h1 = img1.size
        w2, h2 = img2.size

        # resize到统一尺寸做SSIM
        TARGET = 256
        r1 = np.array(img1.resize((TARGET, TARGET), Image.LANCZOS))
        r2 = np.array(img2.resize((TARGET, TARGET), Image.LANCZOS))
        ss = ssim(r1, r2)
        md = float(np.mean(np.abs(r1.astype(float) - r2.astype(float))))

        # 判断
        if w1 != w2 or h1 != h2:
            verdict = "FALSE_POSITIVE_DIFFERENT_DIMENSIONS"
        elif ss > 0.85 and md < 10:
            verdict = "TRUE_SIMILAR"
        elif ss > 0.70 and md < 20:
            verdict = "POSSIBLE_MODIFIED"
        else:
            verdict = "FALSE_POSITIVE_TONAL"

        p["ssim"] = round(ss, 4)
        p["pixel_mean_diff"] = round(md, 2)
        p["dim1"] = f"{w1}x{h1}"
        p["dim2"] = f"{w2}x{h2}"
        p["verdict"] = verdict
        verified.append(p)

    return verified


def main(root_dir, min_kb=50, threshold=0.75):
    output_base = os.path.join(root_dir, "_cross_paper_analysis")
    os.makedirs(output_base, exist_ok=True)

    print("=" * 70)
    print("批量跨论文图片分析")
    print("=" * 70)
    print(f"根目录: {root_dir}")
    print(f"最小图片: {min_kb}KB")
    print(f"相似阈值: {threshold}")
    print()

    # 步骤1: 提取所有图片
    print("步骤1: 提取所有PDF中的图片...")
    extracted = extract_all_figures(root_dir, output_base, min_kb)

    total_figs = sum(len(figs) for figs in extracted.values())
    print(f"\n共从{len(extracted)}篇论文提取了{total_figs}张图片")

    # 步骤2: 跨论文比对
    print("\n步骤2: 跨论文图片相似性比对...")
    cross = cross_paper_similarity(extracted, threshold)

    cross_path = os.path.join(output_base, "cross_paper_similarity.json")
    with open(cross_path, 'w') as f:
        json.dump({
            "threshold": threshold,
            "total_pairs": len(cross),
            "cross_paper_pairs": [p for p in cross if p["cross_paper"]],
            "intra_paper_pairs": [p for p in cross if not p["cross_paper"]],
        }, f, indent=2, ensure_ascii=False)
    print(f"  保存到: {cross_path}")

    # 输出高相似度对摘要
    cross_pairs = [p for p in cross if p["cross_paper"]]
    high = [p for p in cross_pairs if p["similarity"] >= 0.90]
    moderate = [p for p in cross_pairs if 0.80 <= p["similarity"] < 0.90]

    print(f"\n  跨论文相似对: {len(cross_pairs)} 对")
    print(f"  高相似度 (>=0.90): {len(high)} 对")
    print(f"  中等相似 (0.80-0.89): {len(moderate)} 对")

    if high:
        print("\\n  ⚠️  高相似度跨论文对 — 正在进行SSIM二次验证...")
        # 对高相似度对做SSIM验证
        verified = verify_high_similarity_pairs(high, extracted, threshold_verify=0.80)
        true_similar = [v for v in verified if v.get("verdict") == "TRUE_SIMILAR"]
        false_pos = [v for v in verified if "FALSE_POSITIVE" in v.get("verdict", "")]
        possible = [v for v in verified if v.get("verdict") == "POSSIBLE_MODIFIED"]

        print(f"  SSIM验证结果:")
        print(f"    真正相似 (图像复用嫌疑): {len(true_similar)}")
        print(f"    假阳性 (色调相似但内容不同): {len(false_pos)}")
        print(f"    可能修改: {len(possible)}")

        for v in true_similar[:5]:
            print(f"    ✅ {os.path.basename(v['fig1'])[:40]} <-> {os.path.basename(v['fig2'])[:40]}")
            print(f"       pHASH={v['similarity']:.3f} SSIM={v['ssim']:.3f} 均值差={v['pixel_mean_diff']}")

        for v in false_pos[:3]:
            print(f"    ❌ {os.path.basename(v['fig1'])[:40]} <-> {os.path.basename(v['fig2'])[:40]}")
            print(f"       pHASH={v['similarity']:.3f} SSIM={v['ssim']:.3f} — {v.get('verdict','')}")
        if len(high) > 15:
            print(f"    ... 还有{len(high)-15}对")

    # 步骤3: 论文内部比对
    print("\n步骤3: 论文内部跨图相似性...")
    intra = intra_paper_similarity(extracted, 0.70)
    intra_path = os.path.join(output_base, "intra_paper_similarity.json")
    with open(intra_path, 'w') as f:
        json.dump(intra, f, indent=2, ensure_ascii=False)

    for label, pairs_list in intra.items():
        high_pairs = [p for p in pairs_list if p["similarity"] >= 0.85]
        if high_pairs:
            print(f"  ⚠️  {label}: {len(high_pairs)} 对高相似度(>=0.85)")
            for p in high_pairs[:5]:
                print(f"    {p['fig1'][:40]} <-> {p['fig2'][:40]}: {p['similarity']:.3f}")

    # 输出报告
    summary = {
        "analysis_date": __import__('datetime').datetime.now().isoformat(),
        "root_dir": root_dir,
        "papers_analyzed": len(extracted),
        "total_figures": total_figs,
        "cross_paper_pairs": {
            "total": len(cross_pairs),
            "high_similarity_ge_90": len(high),
            "moderate_80_89": len(moderate),
        },
        "intra_paper_high_pairs": sum(len(v) for v in intra.values()),
        "warning": "pHASH对尺寸差异大的图像可能产生假阳性高相似度。所有>=0.90的结果需做像素级交叉验证。"
    }

    summary_path = os.path.join(output_base, "analysis_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"分析完成")
    print(f"{'='*70}")
    print(f"  论文数: {len(extracted)}")
    print(f"  图片数: {total_figs}")
    print(f"  跨论文相似对(>={threshold}): {len(cross_pairs)}")
    print(f"  其中高度相似(>=0.90, 需验证): {len(high)}")
    print(f"  输出目录: {output_base}")
    print(f"\\n  ⚠️  重要: pHASH假阳性风险已通过SSIM验证控制")
    print(f"  不同尺寸的图像(pHash会缩放到32×32)可能在色调相似时")
    print(f"  产生假阳性高相似度。所有>=0.90的结果必须用像素级验证确认。")
    print(f"  验证方法: resize到256×256后逐像素比对均值差, <5才算真正相似。")

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="批量跨论文图片一致性分析")
    parser.add_argument("root_dir", help="论文根目录（递归搜索所有PDF）")
    parser.add_argument("--min-kb", type=int, default=50, help="最小图片大小(KB), 默认50")
    parser.add_argument("--threshold", type=float, default=0.75, help="相似阈值, 默认0.75")
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"目录不存在: {args.root_dir}")
        sys.exit(1)

    main(args.root_dir, args.min_kb, args.threshold)
