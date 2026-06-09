#!/usr/bin/env python3
"""
SIFT/ORB 特征点匹配 — 检测图片旋转/翻转/移位后的重复
===========================================================
文章3 (Nature 2020, Acuna/Bik) 的三类图片造假之「重复定位」:
图像相对另一图像移位/旋转/反转后重新使用。

pHASH 对此类造假完全失效（旋转/翻转后hash值完全改变）。
SIFT/ORB 特征点匹配不受旋转/缩放/仿射变换影响。

⚠️ 关键使用前提：需配合 MinerU bbox 级别裁剪后的单个子图使用。
不要对 PyMuPDF 整页提取的图直接用——会陷入「同一 WB 图的
多个嵌入版本互相高匹配」的假阳性噪音。

用法:
    python sift_duplicate_detection.py img1.png img2.png
    python sift_duplicate_detection.py --dir images/ --min-matches 10
    python sift_duplicate_detection.py --dir images/ --json results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_image(path):
    """加载图片为 numpy 数组（灰度）"""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("L")
        return np.array(img), img.size
    except Exception as e:
        return None, str(e)


def detect_sift_keypoints(img_array):
    """使用 SIFT 检测关键点和描述符"""
    try:
        import cv2
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img_array, None)
        return kp, des
    except ImportError:
        pass
    
    # Fallback: ORB (opencv always has it, no patent issues)
    try:
        import cv2
        orb = cv2.ORB_create(nfeatures=2000)
        kp, des = orb.detectAndCompute(img_array, None)
        return kp, des
    except ImportError:
        return None, "opencv-python not installed"


def match_images(des1, des2, method="FLANN"):
    """匹配两个描述符集"""
    try:
        import cv2
        
        if method == "FLANN" and des1.shape[1] > 20:  # SIFT descriptors are 128-dim
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
        else:
            # Brute force for ORB or fallback
            bf = cv2.BFMatcher(cv2.NORM_HAMMING if des1.dtype == 'uint8' and des1.shape[1] < 50 else cv2.NORM_L2, crossCheck=False)
            raw_matches = bf.knnMatch(des1, des2, k=2)
            # Filter with Lowe's ratio test
            good = []
            for m, n in raw_matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            return len(good), f"BruteForce+Lowe(0.75): {len(good)} good matches"
        
        # Lowe's ratio test for FLANN
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        return len(good_matches), f"FLANN+Lowe(0.75): {len(good_matches)} good matches"
    
    except Exception as e:
        return 0, f"Match error: {e}"


def analyze_pair(path1, path2):
    """分析两张图片是否匹配（含旋转/翻转/缩放不变性）"""
    img1, size1 = load_image(path1)
    img2, size2 = load_image(path2)
    
    if img1 is None or img2 is None:
        error = f"Load error: {size1 if img1 is None else size2}"
        return {"error": error, "match_count": 0, "verdict": "ERROR"}
    
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    
    # Detect keypoints
    kp1, des1 = detect_sift_keypoints(img1)
    kp2, des2 = detect_sift_keypoints(img2)
    
    if kp1 is None or kp2 is None:
        return {"error": "opencv-python not installed", "match_count": 0, "verdict": "ERROR"}
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return {
            "keypoints1": len(kp1), "keypoints2": len(kp2),
            "match_count": 0,
            "verdict": "LOW_TEXTURE",
            "note": "图片纹理不足，无法提取足够特征点"
        }
    
    n_matches, method_desc = match_images(des1, des2)
    
    # 关键过滤：相同整页 WB 图之间有大量背景匹配 → 假阳性
    # 真正的「重复定位」通常涉及不同尺寸/不同背景的图片
    # 过滤规则：
    # 1. 如果两张图片尺寸高度相似(宽高各差<20%)且都是大图(>1000px) → 可能是同级WB板，降权
    # 2. 如果一张图是另一张的子区域 → 真正的复制裁剪，保留
    from collections import namedtuple
    
    size_similarity_w = abs(w1 - w2) / max(w1, w2)
    size_similarity_h = abs(h1 - h2) / max(h1, h2)
    both_large = w1 > 1000 and w2 > 1000 and h1 > 800 and h2 > 800
    similar_size = size_similarity_w < 0.20 and size_similarity_h < 0.20
    
    # 如果两张大图的尺寸高度相似 → WB板级别的布局相似，不是部分复制
    # 需要更高的匹配门槛（至少 50 个匹配点 + 面积也高度重叠）
    if both_large and similar_size:
        effective_threshold = 100  # v8.8.1: 从50升至100，基于阴性对照基线（PLOS ONE 63.3% FPR）
    else:
        # 子区域检测：小图在大图里 → 真正的复制
        area_ratio = (w1 * h1) / (w2 * h2) if w2 * h2 > 0 else 1
        if (area_ratio < 0.3 or area_ratio > 3.0) and both_large:
            effective_threshold = 40  # v8.8.1: 从20升至40
        else:
            effective_threshold = 60  # v8.8.1: 从30升至60
    
    if n_matches >= effective_threshold:
        verdict = "HIGH_MATCH"
        risk = "high"
    elif n_matches >= max(8, effective_threshold // 3):
        verdict = "MODERATE_MATCH"
        risk = "medium"
    elif n_matches >= 4:
        verdict = "LOW_MATCH"
        risk = "low"
    else:
        verdict = "NO_MATCH"
        risk = "clean"
    
    return {
        "file1": Path(path1).name,
        "file2": Path(path2).name,
        "size1": f"{w1}×{h1}",
        "size2": f"{w2}×{h2}",
        "keypoints1": len(kp1),
        "keypoints2": len(kp2),
        "match_count": n_matches,
        "match_method": method_desc,
        "verdict": verdict,
        "risk": risk,
        "effective_threshold": effective_threshold,
        "size_similarity": f"w:{size_similarity_w:.2f} h:{size_similarity_h:.2f}",
        "note": "同级WB版，用high threshold" if both_large and similar_size else "",
    }


def scan_directory(image_dir, min_matches=8):
    """扫描目录中所有图片对"""
    from itertools import combinations
    
    # Find all images
    exts = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    images = []
    for f in sorted(Path(image_dir).iterdir()):
        if f.suffix.lower() in exts:
            images.append(str(f))
    
    if len(images) < 2:
        return {"error": f"Only {len(images)} images found, need >= 2"}
    
    results = []
    total_pairs = len(images) * (len(images) - 1) // 2
    print(f"📂 {image_dir}")
    print(f"   图片数: {len(images)} | 对比对数: {total_pairs}")
    
    suspicious = []
    for i, (p1, p2) in enumerate(combinations(images, 2)):
        if i % 100 == 0:
            print(f"   进度: {i}/{total_pairs}...")
        
        result = analyze_pair(p1, p2)
        if result.get("risk") in ("high", "medium"):
            result["path1"] = p1
            result["path2"] = p2
            suspicious.append(result)
        
        results.append(result)
    
    # Sort by match count descending
    suspicious.sort(key=lambda x: x["match_count"], reverse=True)
    
    return {
        "n_images": len(images),
        "n_pairs": total_pairs,
        "n_suspicious": len(suspicious),
        "suspicious": suspicious,
        "min_matches": min_matches,
    }


def main():
    parser = argparse.ArgumentParser(description="SIFT/ORB 特征点匹配检测图片重复")
    parser.add_argument("image1", nargs="?", help="第一张图片")
    parser.add_argument("image2", nargs="?", help="第二张图片")
    parser.add_argument("--dir", help="批量扫描目录")
    parser.add_argument("--min-matches", type=int, default=8, help="最小匹配点数（默认8）")
    parser.add_argument("--json", help="输出JSON文件")
    args = parser.parse_args()
    
    # 检查 opencv
    try:
        import cv2
    except ImportError:
        print("错误: opencv-python 未安装")
        print("安装: pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple")
        sys.exit(1)
    
    if args.dir:
        print("═" * 64)
        print("  SIFT/ORB 特征点匹配 — 批量扫描")
        print("═" * 64)
        
        result = scan_directory(args.dir, args.min_matches)
        
        if "error" in result:
            print(f"\n错误: {result['error']}")
            sys.exit(1)
        
        print(f"\n{'═'*64}")
        print(f"  扫描完成: {result['n_images']} 张图片, {result['n_pairs']} 对比较")
        print(f"  可疑匹配: {result['n_suspicious']} 对")
        
        if result["suspicious"]:
            print(f"\n  🔴 可疑匹配排名:\n")
            for i, s in enumerate(result["suspicious"]):
                print(f"  #{i+1} {s.get('path1','?').split('/')[-1]} vs {s.get('path2','?').split('/')[-1]}")
                print(f"       匹配点: {s['match_count']} | 判定: {s['verdict']} | 风险: {s['risk']}")
                print(f"       尺寸: {s['size1']} vs {s['size2']}")
                if s['match_count'] >= 15:
                    print(f"       ⚠️ 强烈建议: 运行像素级SSIM验证 + ELA分析三重确认")
                print()
        else:
            print(f"\n  ✅ 未发现可疑的特征点匹配")
        
        if args.json:
            with open(args.json, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"📄 JSON已保存: {args.json}")
    
    else:
        if not args.image1 or not args.image2:
            print("用法: sift_duplicate_detection.py img1.png img2.png")
            print("      sift_duplicate_detection.py --dir images/")
            sys.exit(1)
        
        print("═" * 64)
        print("  SIFT/ORB 特征点匹配 — 图片对检测")
        print("═" * 64)
        
        result = analyze_pair(args.image1, args.image2)
        
        if "error" in result:
            print(f"\n错误: {result['error']}")
            sys.exit(1)
        
        print(f"\n  {result['file1']} ({result['size1']})")
        print(f"  vs")
        print(f"  {result['file2']} ({result['size2']})")
        print(f"\n  关键点: {result['keypoints1']} vs {result['keypoints2']}")
        print(f"  匹配点: {result['match_count']}")
        print(f"  方法: {result['match_method']}")
        print(f"  判定: {result['verdict']} | 风险: {result['risk']}")
        
        if result["risk"] in ("high", "medium"):
            print(f"\n  ⚠️ 建议: 运行像素级SSIM验证 + ELA分析三重确认")


if __name__ == "__main__":
    main()
