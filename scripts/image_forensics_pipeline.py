#!/usr/bin/env python3
"""
论文图片全流程取证分析

从论文PDF提取图片 → 块重复检测 + ELA + 背景分析 + 跨图比对

用法:
  python3 image_forensics_pipeline.py <paper.pdf> [output_dir]

依赖:
  pip3 install pymupdf scipy scikit-image Pillow numpy
"""

import os, sys, json
import numpy as np
from PIL import Image
from scipy.signal import find_peaks
import itertools

def extract_figures(pdf_path, out_dir, min_size=100):
    """从PDF提取所有>=min_size像素的图片"""
    import fitz
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    print(f"PDF: {pdf_path} ({doc.page_count}页)")
    
    extracted = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        images = page.get_images(full=True)
        for idx, img_info in enumerate(images):
            xref = img_info[0]
            base = doc.extract_image(xref)
            w, h = base["width"], base["height"]
            if w < min_size or h < min_size:
                continue
            ext = base["ext"]
            fname = f"page{page_num+1}_{idx+1}_{w}x{h}.{ext}"
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "wb") as f:
                f.write(base["image"])
            extracted.append(fpath)
            print(f"  提取: {fname} ({len(base['image'])//1024}KB)")
    
    doc.close()
    return extracted

def block_duplication_detection(gray, block_size=32, bg_threshold=248):
    """
    内容区域块重复检测（排除纯色背景）
    返回: 重复块数, 总内容块数
    """
    h, w = gray.shape
    
    # 内容边界
    row_var = np.var(gray.astype(float), axis=1)
    col_var = np.var(gray.astype(float), axis=0)
    content_rows = np.where(row_var > 10)[0]
    content_cols = np.where(col_var > 10)[0]
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        return 0, 0
    
    top, bottom = int(content_rows[0]), int(content_rows[-1])
    left, right = int(content_cols[0]), int(content_cols[-1])
    
    blocks = {}
    dups = 0
    total = 0
    
    for y in range(top, bottom - block_size, block_size):
        for x in range(left, right - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size].astype(float)
            bm = np.mean(block)
            if bm > bg_threshold or bm < 10:
                continue
            total += 1
            
            # 均值哈希
            hash_arr = (block > bm).astype(int).flatten()[:64]
            if len(hash_arr) < 64:
                continue
            hval = 0
            for b in hash_arr:
                hval = (hval << 1) | int(b)
            
            if hval in blocks:
                py, px = blocks[hval]
                prev = gray[py:py+block_size, px:px+block_size].astype(float)
                if np.mean(np.abs(block - prev)) < 8:
                    dups += 1
            else:
                blocks[hval] = (y, x)
    
    return dups, total

def ela_analysis(img_path):
    """JPEG误差水平分析 — 返回4象限ELA。使用唯一临时文件避免并行冲突。"""
    img = np.array(Image.open(img_path).convert('RGB'))
    # 使用进程PID+随机后缀避免并行执行时的文件冲突
    temp = f'/tmp/_ela_pipeline_{os.getpid()}_{np.random.randint(0, 99999)}.jpg'
    Image.open(img_path).save(temp, 'JPEG', quality=85)
    re_img = np.array(Image.open(temp).convert('RGB')).astype(float)
    os.remove(temp)
    
    diff = np.abs(img.astype(float) - re_img.astype(float))
    gray_diff = np.mean(diff, axis=2)
    
    h, w = gray_diff.shape
    qh, qw = h // 2, w // 2
    
    q_ela = [
        float(np.mean(gray_diff[:qh, :qw])),
        float(np.mean(gray_diff[:qh, qw:])),
        float(np.mean(gray_diff[qh:, :qw])),
        float(np.mean(gray_diff[qh:, qw:]))
    ]
    ela_range = max(q_ela) - min(q_ela)
    
    return q_ela, round(ela_range, 3)

def background_analysis(gray, step=50):
    """检测背景均匀性"""
    h = gray.shape[0]
    bg_sections = []
    for y in range(0, h, step):
        bg = float(np.mean(gray[y:y+20, :]))
        bg_sections.append(bg)
    
    bg_arr = np.array(bg_sections)
    bg_diffs = np.abs(np.diff(bg_arr))
    abrupt = np.where(bg_diffs > (np.std(bg_diffs) * 3 + np.mean(bg_diffs)))[0]
    
    return {
        "mean": float(np.mean(bg_arr)),
        "std": float(np.std(bg_arr)),
        "range": [float(np.min(bg_arr)), float(np.max(bg_arr))],
        "max_diff_between_segments": float(np.max(bg_diffs)),
        "abrupt_change_segments": [int(a) for a in abrupt.tolist()]
    }

def quick_phash(gray, size=32):
    """快速均值哈希"""
    from PIL import Image
    img = Image.fromarray(gray).resize((size, size), Image.LANCZOS)
    pix = np.array(img)
    bits = (pix.flatten() > np.mean(pix)).astype(int)[:64]
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h

def analyze_figure(fig_path):
    """单张图片全分析"""
    img = Image.open(fig_path).convert('RGB')
    gray = np.array(img.convert('L'))
    w, h = img.size
    fsize = os.path.getsize(fig_path)
    
    result = {
        "file": os.path.basename(fig_path),
        "size": f"{w}x{h}",
        "file_size_kb": fsize // 1024,
        "format": os.path.splitext(fig_path)[1]
    }
    
    # 块重复
    dups, total = block_duplication_detection(gray)
    result["block_analysis"] = {
        "content_blocks": total,
        "duplicate_blocks": dups,
        "duplication_rate": round(dups / max(total, 1), 4)
    }
    
    # ELA
    if result["format"] in ('.jpg', '.jpeg'):
        q_ela, ela_range = ela_analysis(fig_path)
        result["ela"] = {
            "quadrants": q_ela,
            "range": ela_range,
            "suspicious": ela_range > 0.5
        }
    
    # 背景
    result["background"] = background_analysis(gray)
    result["background"]["abrupt_change_count"] = len(result["background"]["abrupt_change_segments"])
    
    # pHash
    result["phash"] = quick_phash(gray)
    
    return result

def cross_compare(analyses):
    """跨图片比对"""
    files = [a["file"] for a in analyses]
    hashes = {a["file"]: a.get("phash", 0) for a in analyses if "phash" in a}
    
    pairs = []
    for f1, f2 in itertools.combinations(files, 2):
        if f1 not in hashes or f2 not in hashes:
            continue
        h1, h2 = hashes[f1], hashes[f2]
        xor_val = h1 ^ h2
        dist = bin(xor_val).count('1')
        sim = round(1.0 - dist / 64.0, 4)
        if sim > 0.85:  # v8.8.1: 从0.65升至0.85，基于阴性对照基线（PLOS ONE 63.3% FPR→目标<10%）
            pairs.append({
                "fig1": f1,
                "fig2": f2,
                "similarity": sim,
                "level": "HIGH" if sim > 0.85 else "MODERATE"
            })
    
    return pairs

def main(pdf_path, out_dir=None):
    """主流程"""
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    if out_dir is None:
        out_dir = f"figures_{basename}"
    
    # 1. 提取图
    print("=" * 60)
    print("步骤1: 提取PDF中的图片")
    print("=" * 60)
    extracted = extract_figures(pdf_path, out_dir)
    print(f"共提取 {len(extracted)} 张图片")
    
    if not extracted:
        print("未找到图片!")
        return
    
    # 2. 逐图分析
    print("\n" + "=" * 60)
    print("步骤2: 逐图取证分析")
    print("=" * 60)
    analyses = []
    for fpath in sorted(extracted):
        fsize = os.path.getsize(fpath) // 1024
        if fsize < 10:  # 跳过小图标
            print(f"  跳过 (<10KB): {os.path.basename(fpath)}")
            continue
        print(f"\n分析: {os.path.basename(fpath)}")
        result = analyze_figure(fpath)
        analyses.append(result)
        
        # 打印摘要
        br = result["block_analysis"]
        dup_pct = br["duplication_rate"] * 100
        print(f"  块重复: {br['duplicate_blocks']}/{br['content_blocks']} ({dup_pct:.1f}%)")
        
        if "ela" in result:
            e = result["ela"]
            ela_flag = "⚠️ 异常" if e["suspicious"] else "正常"
            print(f"  ELA: {e['quadrants']} 范围={e['range']} ({ela_flag})")
        
        bg = result["background"]
        print(f"  背景突变: {bg['abrupt_change_count']} 处")
    
    # 3. 跨图比对
    print("\n" + "=" * 60)
    print("步骤3: 跨图相似性比对")
    print("=" * 60)
    pairs = cross_compare(analyses)
    for p in pairs:
        print(f"  [{p['level']}] {p['fig1']} <-> {p['fig2']}: sim={p['similarity']}")
    if not pairs:
        print("  无显著相似性")
    
    # 4. 输出JSON
    report = {
        "paper": pdf_path,
        "figures_analyzed": len(analyses),
        "figures": analyses,
        "cross_figure_similarity": pairs,
        "summary": {
            "total_suspicious_ela": sum(1 for a in analyses 
                                       if a.get("ela", {}).get("suspicious", False)),
            "total_high_dup": sum(1 for a in analyses 
                                 if a["block_analysis"]["duplication_rate"] > 0.10),
            "total_abrupt_bg": sum(1 for a in analyses 
                                  if a["background"]["abrupt_change_count"] > 0),
            "high_sim_pairs": len(pairs)
        }
    }
    
    report_path = os.path.join(out_dir, "_forensics_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {report_path}")
    
    # 摘要
    print(f"\n{'='*60}")
    print(f"审查摘要")
    print(f"{'='*60}")
    s = report["summary"]
    print(f"  ELA异常(>0.5): {s['total_suspicious_ela']} 张")
    print(f"  高块重复(>10%): {s['total_high_dup']} 张")
    print(f"  背景突变: {s['total_abrupt_bg']} 张")
    print(f"  跨图相似对: {s['high_sim_pairs']} 对")
    print(f"  ⚠️  当图片同时有ELA异常 + 背景突变 + 高块重复时 → 高度可疑")
    
    return report

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)
    
    main(pdf_path, out_dir)
