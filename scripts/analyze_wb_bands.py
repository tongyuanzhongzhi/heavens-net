#!/usr/bin/env python3
"""
Western blot条带互相关分析脚本

检测论文Western blot图中是否存在条带复制/粘贴行为。
分析步骤:
1. 从PDF或图片文件提取WB图
2. 检测水平条带（峰值检测）
3. 提取条带轮廓并计算Pearson互相关
4. 输出NEAR-IDENTICAL条带对（corr>0.95）

用法:
  python3 analyze_wb_bands.py <image_file_or_pdf_path>
  python3 analyze_wb_bands.py paper_figure.png
  python3 analyze_wb_bands.py paper.pdf --page 6   # 从PDF指定页提取

依赖:
  pip3 install pymupdf scipy Pillow numpy
"""

import sys, os, json
import numpy as np
from scipy.signal import find_peaks
from PIL import Image

def extract_img_from_pdf(pdf_path, page_num=0):
    """从PDF指定页提取第一张>=100px的图片"""
    import fitz
    doc = fitz.open(pdf_path)
    if page_num >= doc.page_count:
        print(f"错误: PDF共{doc.page_count}页，指定{page_num}超出范围")
        sys.exit(1)
    page = doc[page_num]
    images = page.get_images(full=True)
    for img in images:
        xref = img[0]
        base = doc.extract_image(xref)
        if base["width"] >= 100 and base["height"] >= 100:
            arr = np.frombuffer(base["image"], dtype=np.uint8)
            from io import BytesIO
            pil_img = Image.open(BytesIO(base["image"])).convert('RGB')
            doc.close()
            return pil_img
    doc.close()
    raise ValueError(f"PDF第{page_num+1}页未找到>=100px的图片")

def analyze_wb(image_path, json_output=None, plot_output=None):
    """主分析函数"""
    # 加载图片
    if image_path.lower().endswith('.pdf'):
        img = extract_img_from_pdf(image_path, page_num=0)
    else:
        img = Image.open(image_path).convert('RGB')
    
    gray = np.array(img.convert('L'))
    w, h = img.size
    print(f"图片: {w}x{h}")
    
    # 计算内容边界（跳过白边）
    row_var = np.var(gray.astype(float), axis=1)
    col_var = np.var(gray.astype(float), axis=0)
    content_rows = np.where(row_var > 10)[0]
    content_cols = np.where(col_var > 10)[0]
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        content_top, content_bottom = 0, h
        content_left, content_right = 0, w
    else:
        content_top, content_bottom = int(content_rows[0]), int(content_rows[-1])
        content_left, content_right = int(content_cols[0]), int(content_cols[-1])
    
    print(f"内容区域: ({content_left},{content_top})-({content_right},{content_bottom})")
    
    # === 条带检测 ===
    row_means = np.mean(gray[:, content_left:content_right], axis=1)
    bg_mean = float(np.mean(row_means))
    
    # 反转（暗条带=高峰值）
    signal = bg_mean - row_means
    signal = np.clip(signal, 0, None)
    
    peaks, props = find_peaks(
        signal,
        height=np.std(signal) * 0.5,
        distance=15,
        width=3
    )
    
    print(f"检测到 {len(peaks)} 个条带")
    
    if len(peaks) < 2:
        print("条带太少，无法做互相关分析")
        return {"error": "too_few_bands", "bands_found": int(len(peaks))}
    
    # === 提取条带轮廓 ===
    band_profiles = {}
    for p in peaks:
        y1 = max(content_top, p - 8)
        y2 = min(content_bottom, p + 8)
        profile = gray[y1:y2, content_left:content_right].astype(float)
        profile = (profile - np.mean(profile)) / (np.std(profile) + 1e-6)
        band_profiles[int(p)] = profile.flatten()
    
    # === 互相关矩阵 ===
    peak_list = sorted(band_profiles.keys())
    results = {
        "image": image_path,
        "size": f"{w}x{h}",
        "content_bbox": [content_left, content_top, content_right, content_bottom],
        "total_bands": len(peak_list),
        "band_pairs": [],
        "near_identical": [],
        "highly_similar": []
    }
    
    print(f"\n条带互相关分析 ({len(peak_list)} 个条带):")
    print(f"{'条带1(行)':<12} {'条带2(行)':<12} {'相关系数':<10} {'判断':<20}")
    print("-" * 55)
    
    for i in range(len(peak_list)):
        for j in range(i + 1, len(peak_list)):
            p1, p2 = peak_list[i], peak_list[j]
            prof1 = band_profiles[p1]
            prof2 = band_profiles[p2]
            
            min_len = min(len(prof1), len(prof2))
            corr = float(np.corrcoef(prof1[:min_len], prof2[:min_len])[0, 1])
            
            if corr > 0.80:
                pair = {
                    "band1_row": p1,
                    "band2_row": p2,
                    "correlation": round(corr, 4),
                    "y_distance": p2 - p1
                }
                results["band_pairs"].append(pair)
                
                if corr > 0.95:
                    results["near_identical"].append(pair)
                    print(f"Row {p1:<5}    Row {p2:<5}    {corr:<8.4f}  *** NEAR-IDENTICAL ***")
                elif corr > 0.90:
                    results["highly_similar"].append(pair)
                    print(f"Row {p1:<5}    Row {p2:<5}    {corr:<8.4f}  ** VERY HIGH **")
                else:
                    print(f"Row {p1:<5}    Row {p2:<5}    {corr:<8.4f}  HIGH")
    
    # === 摘要 ===
    print(f"\n=== 摘要 ===")
    print(f"总条带: {len(peak_list)}")
    print(f"高度相似对(>0.90): {len(results['highly_similar'])}")
    print(f"近全等对(>0.95): {len(results['near_identical'])}")
    
    if results["near_identical"]:
        max_corr = max(p["correlation"] for p in results["near_identical"])
        print(f"** 最大相关系数: {max_corr:.4f}")
        print(f"** 结论: 存在NEAR-IDENTICAL条带对 — 强烈可疑，建议获取原始WB膜验证")
    elif results["highly_similar"]:
        print(f"** 结论: 存在高度相似条带对 — 需确认是否跨panel")
    else:
        print(f"结论: 未发现条带异常")
    
    # === 辅助：背景均匀性 ===
    bg_sections = []
    for y in range(0, h, 50):
        bg_mean_section = float(np.mean(gray[y:y+20, content_left:content_right]))
        bg_sections.append(bg_mean_section)
    bg_arr = np.array(bg_sections)
    bg_diffs = np.abs(np.diff(bg_arr))
    abrupt = np.where(bg_diffs > np.std(bg_arr) * 3)[0]
    
    results["background_analysis"] = {
        "bg_mean": float(np.mean(bg_arr)),
        "bg_std": float(np.std(bg_arr)),
        "bg_range": [float(np.min(bg_arr)), float(np.max(bg_arr))],
        "abrupt_changes": [int(a) for a in abrupt.tolist()]
    }
    
    if len(abrupt) > 0:
        print(f"背景突变位置(50px段): {abrupt.tolist()}")
        print(f"  → 可能的拼接边界")
    
    # 可选保存JSON
    if json_output:
        with open(json_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果保存: {json_output}")
    
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    image_path = sys.argv[1]
    json_output = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        sys.exit(1)
    
    analyze_wb(image_path, json_output)
