"""
基于 Sobel 算子的 LQ 图像纹理区域自动检测脚本。

检测 LQ 图中梯度密集的高频纹理区域（如花边、刺绣、织物纹理等），
输出带框的可视化图和 bbox 坐标文件，可直接用于 test_lace_region_enhance.py。

用法：
    python detect_texture_regions.py val_images/921059926060_lq.jpg
    python examples/qwen_image/detect_texture_regions.py val_images/921059926060_lq.jpg \
    --mask_path val_images/921059926060_lq.png \
    --thresh_ratio 2.0 \
    --min_area 500

    python detect_texture_regions.py val_images/921059926060_ref_crop.png \
    --mask_path val_images/921059926060_ref_crop_mask.png \
    --thresh_ratio 1.5 \
    --min_area 200 \
    --dilate_kernel 8

可选参数：
    --blur_sigma      预处理高斯模糊 sigma（抑制噪声，默认 1.0）
    --thresh_ratio    自适应阈值比例（相对于梯度均值，默认 2.0）
    --min_area        最小连通域面积（过滤噪点，默认 500）
    --max_boxes       最多保留几个框（按梯度强度排序，默认 5）
    --dilate_kernel   形态学膨胀核大小（合并邻近区域，默认 15）
    --output_dir      输出目录（默认当前目录）

输出：
    - detected_regions_vis.png   带框的 LQ 原图可视化
    - detected_bboxes.txt        每行一个 x1,y1,x2,y2
"""
import argparse
import os
import cv2
import numpy as np


def detect_texture_regions(image_path, mask_path=None, blur_sigma=1.0, thresh_ratio=2.0,
                           min_area=500, max_boxes=5, dilate_kernel=15, mask_dilate=20):
    """
    对 LQ 图像进行 Sobel 纹理区域检测。

    Returns:
        bboxes: list of (x1, y1, x2, y2)，按梯度强度降序排列
        vis_img: 带框的可视化图像 (numpy BGR)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 预处理：轻微高斯模糊抑制压缩伪影/噪声
    if blur_sigma > 0:
        ksize = int(blur_sigma * 6) | 1  # 确保奇数
        gray_blurred = cv2.GaussianBlur(gray, (ksize, ksize), blur_sigma)
    else:
        gray_blurred = gray

    # Gabor 纹理能量检测：对周期性纹理（花边/织物）比边缘检测更鲁棒
    # 多方向 + 多频率的 Gabor 滤波器组，取最大响应作为纹理能量图
    h, w = gray_blurred.shape
    texture_energy = np.zeros((h, w), dtype=np.float64)

    # Gabor 参数：多个频率 × 多个方向
    frequencies = [0.05, 0.1, 0.2]  # 归一化频率，覆盖粗到细纹理
    orientations = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # 0°, 45°, 90°, 135°
    gabor_ksize = 31

    for freq in frequencies:
        for theta in orientations:
            kernel = cv2.getGaborKernel(
                (gabor_ksize, gabor_ksize),
                sigma=gabor_ksize / 6.0,
                theta=theta,
                lambd=1.0 / freq,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_64F,
            )
            filtered = cv2.filter2D(gray_blurred, cv2.CV_64F, kernel)
            # 取绝对值作为能量响应
            texture_energy = np.maximum(texture_energy, np.abs(filtered))

    # 局部方差增强：纹理区域方差大，平坦区域方差小
    local_var_win = max(16, min(h, w) // 32)
    local_mean = cv2.blur(texture_energy.astype(np.float32), (local_var_win, local_var_win))
    local_sq_mean = cv2.blur((texture_energy ** 2).astype(np.float32), (local_var_win, local_var_win))
    local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0).astype(np.float64)

    # 融合：Gabor 能量 × 局部方差 → 纹理显著性图
    gradient_magnitude = texture_energy * np.sqrt(local_variance + 1e-8)
    print(f"[Info] Gabor 纹理检测完成 (freqs={frequencies}, orients={len(orientations)}, var_win={local_var_win})")

    # 应用 mask：将 mask 有效区域转为包围 bbox，仅在这些 bbox 内检测
    valid_bboxes = []
    if mask_path is not None:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"无法读取 mask 图像: {mask_path}")
        # resize mask 到与 LQ 相同尺寸
        if mask_img.shape[:2] != gray.shape[:2]:
            mask_img = cv2.resize(mask_img, (gray.shape[1], gray.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        # 二值化
        _, binary_valid = cv2.threshold(mask_img, 0, 255, cv2.THRESH_BINARY)
        # 膨胀 mask：向外扩展，确保花边边缘不被 bbox 边界截断
        if mask_dilate > 0:
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilate, mask_dilate))
            binary_valid = cv2.dilate(binary_valid, dilate_k, iterations=1)
            print(f"[Info] Mask 已膨胀 {mask_dilate}px")
        # 连通域分析得到包围 bbox
        num_valid, _, valid_stats, _ = cv2.connectedComponentsWithStats(binary_valid, connectivity=8)
        for vi in range(1, num_valid):
            vx = valid_stats[vi, cv2.CC_STAT_LEFT]
            vy = valid_stats[vi, cv2.CC_STAT_TOP]
            vw = valid_stats[vi, cv2.CC_STAT_WIDTH]
            vh = valid_stats[vi, cv2.CC_STAT_HEIGHT]
            if vw * vh < min_area:
                continue
            valid_bboxes.append((vx, vy, vx + vw, vy + vh))
        print(f"[Info] Mask 有效区域转为 {len(valid_bboxes)} 个包围 bbox")

        # 将梯度图中所有 bbox 外的区域置零
        roi_mask = np.zeros_like(gradient_magnitude, dtype=np.uint8)
        for (bx1, by1, bx2, by2) in valid_bboxes:
            roi_mask[by1:by2, bx1:bx2] = 255
        gradient_magnitude[roi_mask == 0] = 0.0

    # 自适应阈值：均值 + ratio * 标准差（仅基于有效区域统计）
    if valid_bboxes:
        valid_pixels = gradient_magnitude[roi_mask > 0]
        mean_grad = valid_pixels.mean() if len(valid_pixels) > 0 else 0.0
        std_grad = valid_pixels.std() if len(valid_pixels) > 0 else 0.0
    else:
        mean_grad = gradient_magnitude.mean()
        std_grad = gradient_magnitude.std()

    # 局部自适应阈值：分块计算均值+std，避免强纹理拉高全局阈值导致弱纹理被淹没
    block_size = max(64, min(gray.shape[0], gray.shape[1]) // 8)
    binary_mask = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    h, w = gradient_magnitude.shape
    for by in range(0, h, block_size // 2):
        for bx in range(0, w, block_size // 2):
            y1, y2 = by, min(by + block_size, h)
            x1, x2 = bx, min(bx + block_size, w)
            block = gradient_magnitude[y1:y2, x1:x2]
            if valid_bboxes:
                block_roi = roi_mask[y1:y2, x1:x2]
                valid_block = block[block_roi > 0]
                if len(valid_block) == 0:
                    continue
                local_mean = valid_block.mean()
                local_std = valid_block.std()
            else:
                local_mean = block.mean()
                local_std = block.std()
            local_thresh = local_mean + thresh_ratio * local_std
            binary_mask[y1:y2, x1:x2] = np.maximum(
                binary_mask[y1:y2, x1:x2],
                (block > local_thresh).astype(np.uint8) * 255
            )
    print(f"[Info] 局部自适应阈值完成 (block_size={block_size}, ratio={thresh_ratio})")

    # 形态学操作：闭运算填充缝隙 + 膨胀合并邻近区域
    if dilate_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_kernel, dilate_kernel))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # 过滤小区域，收集候选框及其平均梯度强度
    candidates = []
    for i in range(1, num_labels):  # 跳过背景 label=0
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        # 计算该区域的平均梯度强度（用于排序）
        region_mask = (labels == i)
        avg_gradient = gradient_magnitude[region_mask].mean()

        candidates.append({
            "bbox": (x, y, x + w, y + h),
            "area": area,
            "avg_gradient": avg_gradient,
        })

    # 按平均梯度强度降序排列，取 top-k
    candidates.sort(key=lambda c: c["avg_gradient"], reverse=True)
    selected = candidates[:max_boxes]

    bboxes = [c["bbox"] for c in selected]

    # 绘制可视化
    vis_img = img.copy()
    colors = [
        (0, 255, 0),    # 绿
        (255, 0, 0),    # 蓝
        (0, 0, 255),    # 红
        (255, 255, 0),  # 青
        (0, 255, 255),  # 黄
    ]
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        color = colors[idx % len(colors)]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label = f"#{idx + 1} ({x2 - x1}x{y2 - y1})"
        cv2.putText(vis_img, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return bboxes, vis_img


def main():
    parser = argparse.ArgumentParser(description="LQ 图像纹理区域自动检测")
    parser.add_argument("image_path", type=str, help="LQ 图像路径")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Mask 图像路径（白色/非零为有效检测区域，排除文字等干扰）")
    parser.add_argument("--blur_sigma", type=float, default=1.0,
                        help="预处理高斯模糊 sigma（默认 1.0）")
    parser.add_argument("--thresh_ratio", type=float, default=2.0,
                        help="自适应阈值比例（默认 2.0）")
    parser.add_argument("--min_area", type=int, default=500,
                        help="最小连通域面积（默认 500）")
    parser.add_argument("--max_boxes", type=int, default=5,
                        help="最多保留几个框（默认 5）")
    parser.add_argument("--dilate_kernel", type=int, default=15,
                        help="形态学膨胀核大小（合并邻近区域，默认 15）")
    parser.add_argument("--mask_dilate", type=int, default=20,
                        help="Mask 膨胀像素数（向外扩展避免边缘截断，默认 20）")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="输出目录（默认当前目录）")
    args = parser.parse_args()

    print(f"[Info] 输入图像: {args.image_path}")
    if args.mask_path:
        print(f"[Info] Mask 图像: {args.mask_path}")
    print(f"[Info] 参数: blur_sigma={args.blur_sigma}, thresh_ratio={args.thresh_ratio}, "
          f"min_area={args.min_area}, max_boxes={args.max_boxes}, dilate_kernel={args.dilate_kernel}, "
          f"mask_dilate={args.mask_dilate}")

    bboxes, vis_img = detect_texture_regions(
        args.image_path,
        mask_path=args.mask_path,
        blur_sigma=args.blur_sigma,
        thresh_ratio=args.thresh_ratio,
        min_area=args.min_area,
        max_boxes=args.max_boxes,
        dilate_kernel=args.dilate_kernel,
        mask_dilate=args.mask_dilate,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image_path))[0]

    # 保存可视化
    vis_path = os.path.join(args.output_dir, f"{basename}_detected_regions_vis.png")
    cv2.imwrite(vis_path, vis_img)
    print(f"\n✅ 可视化已保存: {vis_path}")

    # 保存 bbox 文件
    bbox_path = os.path.join(args.output_dir, f"{basename}_detected_bboxes.txt")
    with open(bbox_path, 'w') as f:
        for bbox in bboxes:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
    print(f"✅ BBox 文件已保存: {bbox_path}")

    # 打印结果
    print(f"\n检测到 {len(bboxes)} 个纹理区域:")
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        print(f"  #{idx + 1}: ({x1},{y1})-({x2},{y2}), 尺寸: {x2 - x1}x{y2 - y1}")
        print(f"       --crop_bbox {x1},{y1},{x2},{y2}")

    if len(bboxes) == 0:
        print("\n⚠️  未检测到纹理区域，尝试降低 --thresh_ratio 或 --min_area")


if __name__ == "__main__":
    main()
