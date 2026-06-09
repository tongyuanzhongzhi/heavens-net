#!/usr/bin/env python3
"""
Heaven's Net — AI 生成图片频域检测器 v1.0
==========================================
通过二维傅里叶变换检测 AI 生成图像在频域的伪造指纹。

原理:
  - GAN 生成图: 上采样层产生棋盘格伪影 → 频域出现规则峰值
  - 扩散模型生成图 (SD/DALL-E): 去噪过程留下周期性模式 → 高频区域异常能量集中
  - 真实照片: 频域分布平滑，能量随频率单调衰减

检测特征:
  1. 高频能量占比 — AI 图高频异常偏高（早期 GAN）
  2. 径向频谱峰值 — AI 图在特定频率范围内有离散尖峰（GAN 指纹）
  3. 中频波动方差 — 扩散模型（SD/DALL-E/Flux）在中频区域产生规则波纹
  4. 频谱峰度 — AI 图频谱分布更尖峭（非高斯性）
  5. 环形能量一致性 — 真实照片各方向能量均衡，AI 图有方向性伪影

依赖: numpy, scipy, PIL (Pillow)
      全部已包含在 Heaven's Net requirements.txt 中

用法:
    from ai_image_detector import AIImageDetector
    detector = AIImageDetector()
    result = detector.analyze("image.png")
    # → {"ai_score": 0.73, "verdict": "suspicious", "features": {...}}
"""

import numpy as np
from scipy import fft
from scipy import ndimage
from scipy.stats import kurtosis
from PIL import Image
import json


class AIImageDetector:
    """AI 生成图片频域检测器
    
    纯算法，零模型下载。基于频谱分析检测 GAN/扩散模型的生成指纹。
    适用于 Western Blot、显微图像、流式图等生物医学图像。
    """
    
    # 阈值（可调）
    HIGH_FREQ_THRESHOLD = 0.35      # 高频能量占比 > 35% → 可疑
    PEAK_COUNT_THRESHOLD = 8        # 径向频谱中异常峰值 > 8 个 → 可疑
    KURTOSIS_THRESHOLD = 5.0        # 频谱峰度 > 5.0 → 可疑
    MID_FREQ_WAVE_THRESHOLD = 0.15   # 中频波动方差 > 0.15 → 扩散模型指纹
    RING_ASYMMETRY_THRESHOLD = 0.25   # 环形能量不对称度 > 0.25 → 方向性伪影
    
    def analyze(self, image_path, verbose=False):
        """分析单张图片
        
        Args:
            image_path: 图片路径
            verbose: 是否输出详细特征
            
        Returns:
            dict: {
                "ai_score": float (0-1, 越高越可疑),
                "verdict": "clean" | "suspicious" | "highly_suspicious",
                "features": {...}
            }
        """
        try:
            img = Image.open(image_path).convert('L')  # 灰度
            img_array = np.array(img, dtype=np.float64)
        except Exception as e:
            return {"ai_score": -1, "verdict": "error", "error": str(e)}
        
        # 归一化
        img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-10)
        
        # 1. 二维 FFT 并获取幅度谱
        f_transform = fft.fft2(img_array)
        f_shifted = fft.fftshift(f_transform)
        magnitude = np.abs(f_shifted)
        magnitude_log = np.log1p(magnitude)
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # 2. 高频能量占比
        # 定义高频区域：频谱外围 50%
        y_indices, x_indices = np.ogrid[:h, :w]
        distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        high_freq_mask = distances > (max_dist * 0.5)
        
        total_energy = np.sum(magnitude)
        high_freq_energy = np.sum(magnitude[high_freq_mask])
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        
        # 3. 径向频谱峰值检测（GAN 指纹核心）
        # 沿径向做 1D 平均，找异常尖峰
        num_bins = 100
        bin_edges = np.linspace(0, max_dist, num_bins + 1)
        radial_profile = []
        
        for i in range(num_bins):
            ring_mask = (distances >= bin_edges[i]) & (distances < bin_edges[i+1])
            if np.any(ring_mask):
                radial_profile.append(np.mean(magnitude[ring_mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # 平滑频谱（中值滤波去噪声）
        smoothed = ndimage.median_filter(radial_profile, size=3)
        
        # 计算残差（原始 - 平滑）= 局部峰值
        residual = radial_profile - smoothed
        median_residual = np.median(np.abs(residual))
        
        if median_residual > 0:
            # 找异常峰值：残差超过中位数 N 倍的点
            peak_threshold = median_residual * self.PEAK_HEIGHT_RATIO
            peak_indices = np.where(residual > peak_threshold)[0]
            peak_count = len(peak_indices)
            peak_heights = residual[peak_indices].tolist() if len(peak_indices) > 0 else []
        else:
            peak_count = 0
            peak_heights = []
        
        # 4. 频谱峰度
        # 真实照片的频谱接近高斯分布，AI 图频谱更尖峭
        kurt = kurtosis(magnitude_log.flatten(), fisher=True)
        
        # 3b. 中频波动方差（扩散模型指纹）
        # 扩散模型在中频（ring 20-60）区域有规则波纹
        mid_start, mid_end = 20, 60
        if len(radial_profile) > mid_end:
            mid_slice = radial_profile[mid_start:mid_end]
            # 对中频区域做 detrend（去掉单调下降趋势）
            x_axis = np.arange(len(mid_slice))
            if len(mid_slice) > 2:
                # 线性拟合去趋势
                coeffs = np.polyfit(x_axis, mid_slice, 1)
                trend = np.polyval(coeffs, x_axis)
                detrended = mid_slice - trend
                # 归一化后计算波动方差
                detrended_norm = detrended / (np.mean(np.abs(detrended)) + 1e-10)
                mid_freq_wave = float(np.var(detrended_norm))
            else:
                mid_freq_wave = 0.0
        else:
            mid_freq_wave = 0.0
        
        # 3c. 环形能量一致性
        # 真实照片频域各方向能量均衡; AI 图有方向性伪影（如棋盘格是水平和垂直的）
        num_angles = 36  # 每 10 度一个扇区
        angle_energies = []
        for a in range(num_angles):
            angle_start = a * np.pi / num_angles
            angle_end = (a + 1) * np.pi / num_angles
            # 创建扇形掩码
            angle_mask = np.zeros_like(magnitude, dtype=bool)
            for y in range(h):
                for x in range(w):
                    dy = y - center_y
                    dx = x - center_x
                    if dx == 0 and dy == 0:
                        continue
                    angle = np.arctan2(dy, dx)
                    # 处理环形对称（频谱是对称的，考虑 0 到 pi）
                    if angle < 0:
                        angle += np.pi
                    if angle_start <= angle < angle_end:
                        angle_mask[y, x] = True
            if np.any(angle_mask):
                angle_energies.append(np.sum(magnitude[angle_mask]))
        
        if len(angle_energies) > 0:
            ring_asymmetry = float(np.std(angle_energies) / (np.mean(angle_energies) + 1e-10))
        else:
            ring_asymmetry = 0.0
        
        # 4. 综合评分（5 特征）
        scores = []
        
        # 高频能量
        if high_freq_ratio > self.HIGH_FREQ_THRESHOLD:
            hf_score = min(1.0, (high_freq_ratio - self.HIGH_FREQ_THRESHOLD) / 0.2 + 0.5)
        else:
            hf_score = high_freq_ratio / self.HIGH_FREQ_THRESHOLD * 0.3
        scores.append(hf_score)
        
        # 峰值数量
        if peak_count > self.PEAK_COUNT_THRESHOLD:
            pk_score = min(1.0, (peak_count - self.PEAK_COUNT_THRESHOLD) / 10 + 0.6)
        else:
            pk_score = peak_count / self.PEAK_COUNT_THRESHOLD * 0.3
        scores.append(pk_score)
        
        # 峰度
        if kurt > self.KURTOSIS_THRESHOLD:
            ku_score = min(1.0, (kurt - self.KURTOSIS_THRESHOLD) / 5 + 0.5)
        else:
            ku_score = max(0, kurt / self.KURTOSIS_THRESHOLD * 0.3)
        scores.append(ku_score)
        
        # 中频波动（扩散模型指纹）
        if mid_freq_wave > self.MID_FREQ_WAVE_THRESHOLD:
            mw_score = min(1.0, (mid_freq_wave - self.MID_FREQ_WAVE_THRESHOLD) / 0.2 + 0.5)
        else:
            mw_score = mid_freq_wave / self.MID_FREQ_WAVE_THRESHOLD * 0.3
        scores.append(mw_score)
        
        # 环形不对称
        if ring_asymmetry > self.RING_ASYMMETRY_THRESHOLD:
            ra_score = min(1.0, (ring_asymmetry - self.RING_ASYMMETRY_THRESHOLD) / 0.2 + 0.5)
        else:
            ra_score = ring_asymmetry / self.RING_ASYMMETRY_THRESHOLD * 0.3
        scores.append(ra_score)
        
        ai_score = float(np.mean(scores))
        
        # 判定
        if ai_score >= 0.7:
            verdict = "highly_suspicious"
        elif ai_score >= 0.4:
            verdict = "suspicious"
        else:
            verdict = "clean"
        
        result = {
            "ai_score": round(ai_score, 3),
            "verdict": verdict,
        }
        
        if verbose:
            result["features"] = {
                "high_freq_ratio": round(high_freq_ratio, 3),
                "peak_count": peak_count,
                "peak_heights": [round(h, 3) for h in peak_heights[:10]],
                "kurtosis": round(kurt, 3),
                "mid_freq_wave": round(mid_freq_wave, 3),
                "ring_asymmetry": round(ring_asymmetry, 3),
                "image_shape": [h, w]
            }
        else:
            result["features"] = {
                "high_freq_ratio": round(high_freq_ratio, 3),
                "peak_count": peak_count,
                "kurtosis": round(kurt, 3),
                "mid_freq_wave": round(mid_freq_wave, 3),
                "ring_asymmetry": round(ring_asymmetry, 3)
            }
        
        return result
    
    def analyze_batch(self, image_paths, verbose=False):
        """批量分析
        
        Returns:
            list of dict: 每张图的分析结果，附 filename
        """
        results = []
        for path in image_paths:
            r = self.analyze(path, verbose=verbose)
            r["filename"] = path
            results.append(r)
        return results


def main():
    """CLI 入口"""
    import argparse
    parser = argparse.ArgumentParser(description="Heaven's Net AI 图片频域检测器")
    parser.add_argument("images", nargs="+", help="图片路径（可多个）")
    parser.add_argument("--verbose", "-v", action="store_true", help="输出详细特征")
    parser.add_argument("--format", choices=["json", "text"], default="text", help="输出格式")
    args = parser.parse_args()
    
    detector = AIImageDetector()
    results = detector.analyze_batch(args.images, verbose=args.verbose)
    
    if args.format == "json":
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for r in results:
            icon = {"clean": "✅", "suspicious": "⚠️", "highly_suspicious": "🚨", "error": "❌"}
            print(f"{icon.get(r.get('verdict'), '?')} {r['filename']}")
            print(f"   AI score: {r.get('ai_score', 'N/A')}  |  verdict: {r.get('verdict', 'N/A')}")
            if r.get('features'):
                feats = r['features']
                print(f"   高频能量: {feats.get('high_freq_ratio', '?')}  |  峰值数: {feats.get('peak_count', '?')}  |  峰度: {feats.get('kurtosis', '?')}")
                print(f"   中频波动: {feats.get('mid_freq_wave', '?')}  |  环形不对称: {feats.get('ring_asymmetry', '?')}")


if __name__ == "__main__":
    main()
