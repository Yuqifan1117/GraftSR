"""
验证实验：分析 test_sr_edit 常规增强结果与 LQ 的残差，
判断是否能通过 LQ 与增强结果的运算提取出"未增强好"的区域（仍模糊/有伪影）。

支持多种域的分析方法：
1. 像素域残差 (Pixel Residual)
2. 频域高频能量差 (Frequency Domain)
3. Laplacian 边缘响应差 (Edge Response)
4. 梯度幅值差 (Gradient Magnitude)
5. 局部方差差 (Local Variance / Texture Activity)

用法:
    python analyze_residual_regions.py \
        --lq_dir /path/to/lq_images \
        --sr_dir /path/to/test_sr_edit_output \
        --gt_dir /path/to/gt_images \       # 可选，用于定量验证
        --output_dir ./residual_analysis \
        --methods pixel,freq,laplacian,gradient,local_var \
        --threshold_mode adaptive \
        --num_samples 20
"""
import os
import io
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_image_as_float(image_path):
    """加载图片并转为 [0,1] float32 numpy array (H,W,3)"""
    img = Image.open(image_path).convert('RGB')
    return np.asarray(img).astype(np.float32) / 255.0


def to_grayscale(rgb_float):
    """RGB [0,1] -> Gray [0,1], shape (H,W)"""
    return 0.2989 * rgb_float[:, :, 0] + 0.5870 * rgb_float[:, :, 1] + 0.1140 * rgb_float[:, :, 2]


# ============================================================
# 各种域的分析方法
# ============================================================

def compute_pixel_residual(lq_gray, sr_gray):
    """像素域绝对残差"""
    return np.abs(sr_gray - lq_gray)


def compute_freq_residual(lq_gray, sr_gray):
    """
    频域高频能量差：
    用 FFT 提取高频分量，比较 SR 和 LQ 的高频能量差异。
    如果 SR 的高频能量没有显著高于 LQ → 该区域未增强好。
    """
    h, w = lq_gray.shape
    
    # FFT
    lq_fft = np.fft.fftshift(np.fft.fft2(lq_gray))
    sr_fft = np.fft.fftshift(np.fft.fft2(sr_gray))
    
    # 构造高通掩膜（中心低频区域置零）
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6  # 保留中心 1/3 为低频
    y_grid, x_grid = np.ogrid[:h, :w]
    lowpass_mask = ((y_grid - cy) ** 2 + (x_grid - cx) ** 2) <= radius ** 2
    highpass_mask = ~lowpass_mask
    
    # 高频能量
    lq_high_freq_energy = np.abs(lq_fft * highpass_mask)
    sr_high_freq_energy = np.abs(sr_fft * highpass_mask)
    
    # 高频能量增益图：SR 比 LQ 多了多少高频
    freq_gain = sr_high_freq_energy - lq_high_freq_energy
    
    # 归一化到 [0,1]
    freq_gain = np.clip(freq_gain, 0, None)
    if freq_gain.max() > 0:
        freq_gain = freq_gain / freq_gain.max()
    
    return freq_gain.astype(np.float32)


def compute_laplacian_residual(lq_gray, sr_gray):
    """
    Laplacian 边缘响应差：
    Laplacian 值大 = 边缘/纹理丰富。
    SR 的 Laplacian 应显著大于 LQ，否则该区域未增强好。
    """
    from scipy.ndimage import laplace
    
    lq_lap = np.abs(laplace(lq_gray))
    sr_lap = np.abs(laplace(sr_gray))
    
    # 边缘响应增益
    lap_gain = sr_lap - lq_lap
    lap_gain = np.clip(lap_gain, 0, None)
    
    if lap_gain.max() > 0:
        lap_gain = lap_gain / lap_gain.max()
    
    return lap_gain.astype(np.float32)


def compute_gradient_residual(lq_gray, sr_gray):
    """
    梯度幅值差（Sobel）：
    类似 Laplacian，但更鲁棒。
    """
    from scipy.ndimage import sobel
    
    lq_gx = sobel(lq_gray, axis=1)
    lq_gy = sobel(lq_gray, axis=0)
    lq_grad_mag = np.sqrt(lq_gx ** 2 + lq_gy ** 2)
    
    sr_gx = sobel(sr_gray, axis=1)
    sr_gy = sobel(sr_gray, axis=0)
    sr_grad_mag = np.sqrt(sr_gx ** 2 + sr_gy ** 2)
    
    grad_gain = sr_grad_mag - lq_grad_mag
    grad_gain = np.clip(grad_gain, 0, None)
    
    if grad_gain.max() > 0:
        grad_gain = grad_gain / grad_gain.max()
    
    return grad_gain.astype(np.float32)


def compute_local_variance_residual(lq_gray, sr_gray, window_size=16):
    """
    局部方差差（纹理活跃度）：
    局部方差大 = 纹理丰富。SR 应在纹理区域有更高的局部方差。
    平坦区域方差低 → 不应被选为二次增强目标。
    """
    from scipy.ndimage import uniform_filter
    
    def local_variance(gray, win):
        mean = uniform_filter(gray, size=win)
        mean_sq = uniform_filter(gray ** 2, size=win)
        var = mean_sq - mean ** 2
        return np.clip(var, 0, None)
    
    lq_var = local_variance(lq_gray, window_size)
    sr_var = local_variance(sr_gray, window_size)
    
    var_gain = sr_var - lq_var
    var_gain = np.clip(var_gain, 0, None)
    
    if var_gain.max() > 0:
        var_gain = var_gain / var_gain.max()
    
    return var_gain.astype(np.float32)


METHOD_REGISTRY = {
    'pixel': compute_pixel_residual,
    'freq': compute_freq_residual,
    'laplacian': compute_laplacian_residual,
    'gradient': compute_gradient_residual,
    'local_var': compute_local_variance_residual,
}


# ============================================================
# 阈值策略
# ============================================================

def compute_adaptive_threshold(residual_map, percentile=70):
    """自适应阈值：取残差图的百分位数作为阈值"""
    return np.percentile(residual_map[residual_map > 0], percentile) if (residual_map > 0).any() else 0.0


def compute_otsu_threshold(residual_map):
    """Otsu 自动阈值"""
    from skimage.filters import threshold_otsu
    try:
        return threshold_otsu(residual_map)
    except Exception:
        return compute_adaptive_threshold(residual_map)


# ============================================================
# 可视化 & 保存
# ============================================================

def save_heatmap(residual_map, output_path):
    """将 [0,1] 残差图保存为真正的伪彩色热力图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=100)
    im = ax.imshow(residual_map, cmap='hot', vmin=0, vmax=1)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Residual Intensity')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def compute_lq_high_freq_mask(lq_gray, method='laplacian', percentile=60):
    """
    从 LQ 中提取高频区域作为"可信信息上界"。
    如果某个区域 LQ 本身就是平坦/模糊的（高频响应极低），
    那即使残差 mask 选中了它，二次增强大概率也是 hallucination，应该抑制。

    Args:
        lq_gray: LQ 灰度图 [0,1], shape (H,W)
        method: 'laplacian' | 'gradient'
        percentile: 高频区域的百分位阈值（LQ 建议比 GT 低，如60）

    Returns:
        binary mask [0,1], shape (H,W)
    """
    if method == 'laplacian':
        from scipy.ndimage import laplace
        response = np.abs(laplace(lq_gray))
    elif method == 'gradient':
        from scipy.ndimage import sobel
        gx = sobel(lq_gray, axis=1)
        gy = sobel(lq_gray, axis=0)
        response = np.sqrt(gx ** 2 + gy ** 2)
    else:
        raise ValueError(f"Unknown LQ high-freq method: {method}")

    thresh = np.percentile(response[response > 0], percentile) if (response > 0).any() else 0
    return (response > thresh).astype(np.float32)


def create_comparison_canvas(lq_pil, sr_pil, residual_maps, binary_masks,
                              method_names, gt_pil=None, lq_hf_mask=None):
    """
    创建对比画布（含伪彩色热力图 + LQ/GT高频区域对照）：
    Row 0: LQ | SR | GT(optional) | LQ_HighFreq(optional)
    Row 1..N: 各方法的残差热力图（伪彩色）
    Row N+1..2N: 各方法的二值 mask
    """
    target_w, target_h = sr_pil.size
    num_methods = len(method_names)

    # Row 0: 原始图像 + 高频参考
    images_row0 = [
        lq_pil.resize((target_w, target_h), Image.BICUBIC),
        sr_pil.resize((target_w, target_h), Image.BICUBIC),
    ]
    if gt_pil is not None:
        images_row0.append(gt_pil.resize((target_w, target_h), Image.BICUBIC))
    if lq_hf_mask is not None:
        hf_uint8 = (lq_hf_mask * 255).astype(np.uint8)
        hf_pil = Image.fromarray(hf_uint8).convert('RGB').resize((target_w, target_h), Image.NEAREST)
        images_row0.append(hf_pil)

    cols = max(len(images_row0), 4)
    rows = 1 + num_methods + num_methods  # row0 + heatmaps + masks

    canvas = Image.new('RGB', (target_w * cols, target_h * rows))

    # Row 0
    for i, img in enumerate(images_row0):
        canvas.paste(img, (i * target_w, 0))

    # Heatmap rows（用 matplotlib 渲染伪彩色后贴回 PIL）
    for j, rmap in enumerate(residual_maps):
        fig, ax = plt.subplots(1, 1, figsize=(target_w / 100, target_h / 100), dpi=100)
        ax.imshow(rmap, cmap='hot', vmin=0, vmax=1)
        ax.axis('off')
        plt.tight_layout(pad=0)
        buf_io = io.BytesIO()
        fig.savefig(buf_io, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        buf_io.seek(0)
        heatmap_pil = Image.open(buf_io).convert('RGB').resize((target_w, target_h), Image.BILINEAR)
        canvas.paste(heatmap_pil, (0, (j + 1) * target_h))

    # Binary mask rows
    for j, bmask in enumerate(binary_masks):
        mask_uint8 = (bmask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8).convert('RGB').resize((target_w, target_h), Image.NEAREST)
        canvas.paste(mask_pil, (0, (j + 1 + num_methods) * target_h))

    return canvas


# ============================================================
# 主流程
# ============================================================

def analyze_single(lq_path, sr_path, output_subdir, methods, threshold_mode, gt_path=None):
    """对单张图执行所有分析方法，并与 LQ 高频区域做 Reliability 验证"""
    lq_rgb = load_image_as_float(lq_path)
    sr_rgb = load_image_as_float(sr_path)

    # 确保尺寸一致
    if lq_rgb.shape != sr_rgb.shape:
        sr_pil = Image.open(sr_path).convert('RGB').resize(
            (lq_rgb.shape[1], lq_rgb.shape[0]), Image.BICUBIC
        )
        sr_rgb = np.asarray(sr_pil).astype(np.float32) / 255.0

    lq_gray = to_grayscale(lq_rgb)
    sr_gray = to_grayscale(sr_rgb)

    gt_pil = None
    if gt_path and os.path.exists(gt_path):
        gt_pil = Image.open(gt_path).convert('RGB')

    lq_pil = Image.open(lq_path).convert('RGB')
    sr_pil = Image.open(sr_path).convert('RGB')

    # 用 LQ 自身的高频区域作为"可信信息上界"
    lq_hf_mask = compute_lq_high_freq_mask(lq_gray, method='laplacian', percentile=60)
    lq_hf_uint8 = (lq_hf_mask * 255).astype(np.uint8)
    Image.fromarray(lq_hf_uint8).save(os.path.join(output_subdir, "lq_high_freq_mask.png"))
    save_heatmap(lq_hf_mask, os.path.join(output_subdir, "lq_high_freq_heatmap.png"))

    residual_maps = []
    binary_masks = []
    valid_method_names = []

    for method_name in methods:
        if method_name not in METHOD_REGISTRY:
            print(f"  [WARN] Unknown method '{method_name}', skipping")
            continue

        func = METHOD_REGISTRY[method_name]
        rmap = func(lq_gray, sr_gray)

        # 阈值化
        if threshold_mode == 'adaptive':
            thresh = compute_adaptive_threshold(rmap, percentile=70)
        elif threshold_mode == 'otsu':
            thresh = compute_otsu_threshold(rmap)
        else:
            thresh = float(threshold_mode)

        bmask = (rmap > thresh).astype(np.float32)

        # 保存单方法结果
        save_heatmap(rmap, os.path.join(output_subdir, f"{method_name}_residual.png"))
        mask_uint8 = (bmask * 255).astype(np.uint8)
        Image.fromarray(mask_uint8).save(os.path.join(output_subdir, f"{method_name}_mask.png"))

        residual_maps.append(rmap)
        binary_masks.append(bmask)
        valid_method_names.append(method_name)

        coverage = bmask.mean() * 100
        print(f"  [{method_name}] threshold={thresh:.4f}, coverage={coverage:.1f}%")

    # 计算每个方法 mask 与 LQ 高频区域的 Reliability
    reliability_results = {}
    for name, bmask in zip(valid_method_names, binary_masks):
        intersection = (bmask * lq_hf_mask).sum()
        detected_area = bmask.sum() + 1e-8
        reliability = intersection / detected_area
        reliability_results[name] = reliability
        print(f"  [{name}] Reliability (overlap with LQ HF): {reliability:.4f}")

    # 融合 mask（取并集）
    if binary_masks:
        fused_mask = np.clip(sum(binary_masks), 0, 1)
        fused_uint8 = (fused_mask * 255).astype(np.uint8)
        Image.fromarray(fused_uint8).save(os.path.join(output_subdir, "fused_mask.png"))
        fused_coverage = fused_mask.mean() * 100
        print(f"  [FUSED] coverage={fused_coverage:.1f}%")

        # Fused mask 的 Reliability
        fused_intersection = (fused_mask * lq_hf_mask).sum()
        fused_reliability = fused_intersection / (fused_mask.sum() + 1e-8)
        reliability_results['fused'] = fused_reliability
        print(f"  [FUSED] Reliability (overlap with LQ HF): {fused_reliability:.4f}")

    # 生成对比画布（传入 lq_hf_mask）
    canvas = create_comparison_canvas(
        lq_pil, sr_pil, residual_maps, binary_masks,
        valid_method_names, gt_pil=gt_pil, lq_hf_mask=lq_hf_mask
    )
    canvas.save(os.path.join(output_subdir, "comparison.png"))

    coverages = {name: mask.mean() for name, mask in zip(valid_method_names, binary_masks)}
    return coverages, reliability_results


def main():
    parser = argparse.ArgumentParser(description="Analyze residual regions between LQ and SR results")
    parser.add_argument("--lq_dir", type=str, required=True, help="Directory of LQ images")
    parser.add_argument("--sr_dir", type=str, required=True, help="Directory of SR output images (from test_sr_edit)")
    parser.add_argument("--gt_dir", type=str, default=None, help="Directory of GT images (optional)")
    parser.add_argument("--output_dir", type=str, default="./residual_analysis", help="Output directory")
    parser.add_argument("--methods", type=str, default="pixel,freq,laplacian,gradient,local_var",
                        help="Comma-separated analysis methods")
    parser.add_argument("--threshold_mode", type=str, default="adaptive",
                        help="Threshold strategy: adaptive, otsu, or a float value")
    parser.add_argument("--num_samples", type=int, default=20, help="Max number of images to analyze")
    args = parser.parse_args()
    
    methods = [m.strip() for m in args.methods.split(',')]
    
    # 收集配对文件
    sr_files = sorted(Path(args.sr_dir).glob("*.png"))
    pairs = []
    for sr_file in sr_files:
        stem = sr_file.stem
        lq_file = Path(args.lq_dir) / f"{stem}.png"
        if not lq_file.exists():
            # 尝试其他常见命名
            for suffix in ['.jpg', '.jpeg', '_lq.png', '_LR.png']:
                alt = Path(args.lq_dir) / f"{stem}{suffix}"
                if alt.exists():
                    lq_file = alt
                    break
        
        if lq_file.exists():
            gt_file = Path(args.gt_dir) / f"{stem}.png" if args.gt_dir else None
            pairs.append((str(lq_file), str(sr_file), str(gt_file) if gt_file and gt_file.exists() else None))
    
    pairs = pairs[:args.num_samples]
    print(f"Found {len(pairs)} image pairs to analyze")
    print(f"Methods: {methods}")
    print(f"Threshold mode: {args.threshold_mode}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_coverages = {m: [] for m in methods}
    all_reliabilities = {m: [] for m in methods}
    all_reliabilities['fused'] = []

    for idx, (lq_path, sr_path, gt_path) in enumerate(pairs):
        filename = Path(sr_path).stem
        print(f"\n[{idx+1}/{len(pairs)}] {filename}")

        output_subdir = os.path.join(args.output_dir, filename)
        os.makedirs(output_subdir, exist_ok=True)

        coverages, reliability_results = analyze_single(
            lq_path, sr_path, output_subdir, methods, args.threshold_mode, gt_path
        )
        for method_name, cov in coverages.items():
            all_coverages[method_name].append(cov)
        for method_name, rel in reliability_results.items():
            if method_name in all_reliabilities:
                all_reliabilities[method_name].append(rel)

    # 汇总统计
    print("\n" + "=" * 60)
    print("SUMMARY: Average coverage (%) of detected refinement regions")
    print("=" * 60)
    for method_name in methods:
        vals = all_coverages.get(method_name, [])
        if vals:
            avg_cov = np.mean(vals) * 100
            std_cov = np.std(vals) * 100
            print(f"  {method_name:12s}: {avg_cov:5.1f}% ± {std_cov:5.1f}%")
        else:
            print(f"  {method_name:12s}: N/A")

    print("\n" + "=" * 60)
    print("SUMMARY: Average Reliability (overlap with LQ high-freq region)")
    print("=" * 60)
    for method_name in methods + ['fused']:
        vals = all_reliabilities.get(method_name, [])
        if vals:
            avg_rel = np.mean(vals)
            std_rel = np.std(vals)
            print(f"  {method_name:12s}: {avg_rel:.4f} ± {std_rel:.4f}")
        else:
            print(f"  {method_name:12s}: N/A")

    # 保存汇总
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Residual Analysis Summary\n")
        f.write(f"Methods: {methods}\n")
        f.write(f"Threshold: {args.threshold_mode}\n")
        f.write(f"Samples: {len(pairs)}\n\n")
        f.write("=== Coverage ===\n")
        for method_name in methods:
            vals = all_coverages.get(method_name, [])
            if vals:
                f.write(f"{method_name}: avg={np.mean(vals)*100:.1f}%, std={np.std(vals)*100:.1f}%\n")
        f.write("\n=== Reliability (overlap with LQ HF) ===\n")
        for method_name in methods + ['fused']:
            vals = all_reliabilities.get(method_name, [])
            if vals:
                f.write(f"{method_name}: avg={np.mean(vals):.4f}, std={np.std(vals):.4f}\n")
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()