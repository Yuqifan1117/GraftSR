#!/usr/bin/env python3
"""
REF 图对齐工具

通过 VLM 双图联合 grounding 识别 HQ 图和 REF 图中同一商品的对应区域，
再用仿射变换将 REF 图变换到与 HQ 图尺寸一致、且商品框完全对齐的状态。

适用场景：一张图只有一个商品框。

用法示例：
    python align_ref_to_hq.py \
        --hq-txt path/to/hq_ori.txt \
        --ref-txt path/to/ref_ori.txt \
        --prompt-txt path/to/prompt_crop.txt \
        --output-dir path/to/test_dataset/ \
        --save-comparison
"""
import argparse
import json
import logging
import os
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

from prompts import PAIRED_GROUNDING_PROMPT
from client import Client, parse_json
from convert_utils import write_image_paths_to_file
from image_utils import pil_image_to_base64, resize_vlm_bboxes

logger = logging.getLogger(__name__)


def optimal_interpolation(scale: float) -> int:
    """
    根据缩放方向选择最优的 OpenCV 插值方法。

    - 缩小（scale < 1.0）：使用 INTER_AREA（像素面积平均），有效抑制锯齿和摩尔纹
    - 放大（scale >= 1.0）：使用 INTER_LANCZOS4（8x8 Lanczos 核），最大程度保留细节

    Args:
        scale: 缩放因子，< 1.0 表示缩小，>= 1.0 表示放大

    Returns:
        OpenCV 插值标志常量
    """
    return cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4


def compute_affine_from_bboxes(src_bbox: List[int],
                               ref_bbox: List[int],
                               max_aspect_ratio_change: float = 0.0,
                               max_scale: float = 0.0,
                               min_scale: float = 0.0) -> np.ndarray:
    """
    根据 SRC 和 REF 的商品框计算仿射变换矩阵（REF → SRC）。

    使用框的左上、右上、左下三个角点作为对应点，
    支持平移、缩放和轻微旋转（对 axis-aligned 框等价于独立 x/y 缩放 + 平移）。

    当 max_aspect_ratio_change > 0 时，限制 x/y 缩放比的比值不超过该阈值，
    防止商品宽高比过度变形。超出阈值时，将两个缩放比向几何均值靠拢，
    使 max(s_x/s_y, s_y/s_x) 刚好等于阈值，框中心仍然对齐。

    Args:
        src_bbox: SRC 图中商品框 [x1, y1, x2, y2]
        ref_bbox: REF 图中商品框 [x1, y1, x2, y2]
        max_aspect_ratio_change: x/y 缩放比的比值上限（如 1.5 表示允许 50% 的宽高比变化）。
                                 设为 0 或负数表示不限制（默认不限制）。
        max_scale: 缩放倍率上限（如 3.0 表示最多放大 3 倍），防止 REF 图被过度放大导致模糊。
                   设为 0 或负数表示不限制（默认不限制）。
        min_scale: 缩放倍率下限（如 0.3 表示最少缩小到 0.3 倍），防止 REF 图被过度缩小丢失细节。
                   设为 0 或负数表示不限制（默认不限制）。

    Returns:
        2x3 仿射变换矩阵
    """
    ref_width = ref_bbox[2] - ref_bbox[0]
    ref_height = ref_bbox[3] - ref_bbox[1]
    src_width = src_bbox[2] - src_bbox[0]
    src_height = src_bbox[3] - src_bbox[1]

    scale_x = src_width / ref_width if ref_width > 0 else 1.0
    scale_y = src_height / ref_height if ref_height > 0 else 1.0

    # 限制缩放倍率的绝对值范围
    if max_scale > 0:
        if scale_x > max_scale or scale_y > max_scale:
            logger.info(f"缩放倍率 s_x={scale_x:.3f}, s_y={scale_y:.3f} 超过上限 {max_scale:.2f}，已裁剪")
            scale_x = min(scale_x, max_scale)
            scale_y = min(scale_y, max_scale)
    if min_scale > 0:
        if scale_x < min_scale or scale_y < min_scale:
            logger.info(f"缩放倍率 s_x={scale_x:.3f}, s_y={scale_y:.3f} 低于下限 {min_scale:.2f}，已裁剪")
            scale_x = max(scale_x, min_scale)
            scale_y = max(scale_y, min_scale)

    if max_aspect_ratio_change > 0 and scale_x > 0 and scale_y > 0:
        ratio = max(scale_x / scale_y, scale_y / scale_x)
        if ratio > max_aspect_ratio_change:
            # 保持几何均值不变，将两个缩放比拉近到阈值范围内
            geometric_mean = np.sqrt(scale_x * scale_y)
            sqrt_limit = np.sqrt(max_aspect_ratio_change)
            if scale_x > scale_y:
                scale_x = geometric_mean * sqrt_limit
                scale_y = geometric_mean / sqrt_limit
            else:
                scale_x = geometric_mean / sqrt_limit
                scale_y = geometric_mean * sqrt_limit
            logger.info(f"宽高比变化 {ratio:.2f} 超过阈值 {max_aspect_ratio_change:.2f}，"
                        f"已调整缩放比为 s_x={scale_x:.3f}, s_y={scale_y:.3f}")

    # 用调整后的缩放比构造目标点（以 SRC 框中心为锚点）
    src_center_x = (src_bbox[0] + src_bbox[2]) / 2.0
    src_center_y = (src_bbox[1] + src_bbox[3]) / 2.0
    adjusted_src_half_w = ref_width * scale_x / 2.0
    adjusted_src_half_h = ref_height * scale_y / 2.0

    src_points = np.float32([
        [src_center_x - adjusted_src_half_w, src_center_y - adjusted_src_half_h],  # 左上
        [src_center_x + adjusted_src_half_w, src_center_y - adjusted_src_half_h],  # 右上
        [src_center_x - adjusted_src_half_w, src_center_y + adjusted_src_half_h],  # 左下
    ])
    ref_points = np.float32([
        [ref_bbox[0], ref_bbox[1]],  # 左上
        [ref_bbox[2], ref_bbox[1]],  # 右上
        [ref_bbox[0], ref_bbox[3]],  # 左下
    ])
    affine_matrix = cv2.getAffineTransform(ref_points, src_points)
    return affine_matrix


def detect_background_color(image: np.ndarray, sample_size: int = 10) -> Tuple[int, ...]:
    """
    检测图像的背景色，取四个角区域像素的中位数。

    Args:
        image: 输入图像 (numpy array, BGR)
        sample_size: 每个角采样的像素区域大小

    Returns:
        背景色 (B, G, R) 元组
    """
    height, width = image.shape[:2]
    corner_size = min(sample_size, height // 4, width // 4, 1)

    corners = [
        image[:corner_size, :corner_size],                          # 左上
        image[:corner_size, width - corner_size:],                  # 右上
        image[height - corner_size:, :corner_size],                 # 左下
        image[height - corner_size:, width - corner_size:],         # 右下
    ]
    all_pixels = np.concatenate([c.reshape(-1, image.shape[2]) for c in corners], axis=0)
    background_color = tuple(int(v) for v in np.median(all_pixels, axis=0))
    return background_color


def align_ref_to_hq(ref_image: np.ndarray,
                    src_size: Tuple[int, int],
                    src_bbox: List[int],
                    ref_bbox: List[int],
                    border_mode: str = "reflect",
                    max_aspect_ratio_change: float = 0.0,
                    max_scale: float = 0.0,
                    min_scale: float = 0.0) -> np.ndarray:
    """
    将 REF 图通过仿射变换对齐到 SRC 图，使商品框完全重合。

    Args:
        ref_image: REF 图像 (numpy array, BGR 或 RGB)
        src_size: SRC 图尺寸 (width, height)
        src_bbox: SRC 图中商品框 [x1, y1, x2, y2]
        ref_bbox: REF 图中商品框 [x1, y1, x2, y2]
        border_mode: 边界填充方式，可选：
            - "reflect": 镜像填充
            - "constant": 黑色填充
            - "replicate": 复制边缘像素
            - "background": 自动检测 REF 图背景色填充
        max_aspect_ratio_change: x/y 缩放比的比值上限，0 表示不限制
        max_scale: 缩放倍率上限，0 表示不限制
        min_scale: 缩放倍率下限，0 表示不限制

    Returns:
        对齐后的 REF 图像，尺寸与 SRC 图一致
    """
    border_modes = {
        "reflect": cv2.BORDER_REFLECT_101,
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "background": cv2.BORDER_CONSTANT,
    }
    cv2_border = border_modes.get(border_mode, cv2.BORDER_REFLECT_101)

    # 检测背景色（仅 background 模式）
    border_value = (0, 0, 0)
    if border_mode == "background":
        border_value = detect_background_color(ref_image)
        logger.info(f"检测到 REF 图背景色 (BGR): {border_value}")

    affine_matrix = compute_affine_from_bboxes(src_bbox, ref_bbox, max_aspect_ratio_change,
                                               max_scale, min_scale)
    src_width, src_height = src_size

    # 根据缩放方向选择最优插值：用几何均值判断整体是放大还是缩小
    ref_box_w = ref_bbox[2] - ref_bbox[0]
    ref_box_h = ref_bbox[3] - ref_bbox[1]
    src_box_w = src_bbox[2] - src_bbox[0]
    src_box_h = src_bbox[3] - src_bbox[1]
    geometric_scale = float(np.sqrt(
        (src_box_w / ref_box_w if ref_box_w > 0 else 1.0) *
        (src_box_h / ref_box_h if ref_box_h > 0 else 1.0)
    ))
    interp_flag = optimal_interpolation(geometric_scale)

    aligned = cv2.warpAffine(
        ref_image,
        affine_matrix,
        (src_width, src_height),
        flags=interp_flag,
        borderMode=cv2_border,
        borderValue=border_value,
    )
    return aligned



def align_ref_to_hq_lossless(ref_image: np.ndarray,
                              src_size: Tuple[int, int],
                              src_bbox: List[int],
                              ref_bbox: List[int],
                              border_mode: str = "constant",
                              max_scale: float = 0.0,
                              min_scale: float = 0.0) -> np.ndarray:
    """
    宽松无损对齐模式：仅通过等比缩放 + 平移 + padding/crop 将 REF 图对齐到 SRC 图。

    与标准仿射对齐的区别：
    - 只做等比缩放（uniform scaling），不改变商品宽高比
    - 商品框中心对齐，大小通过等比缩放尽量匹配
    - 超出画布的部分裁掉，不足的部分用 padding 填充
    - 即使框大小不完全一致也不强制对齐

    对齐策略：
    1. 根据 SRC 和 REF 商品框大小计算等比缩放因子（取 x/y 方向的几何均值）
    2. 应用 max_scale / min_scale 限制
    3. 对 REF 图做等比缩放（使用高质量 Lanczos 插值）
    4. 平移使商品框中心对齐
    5. padding/crop 将画布调整到 SRC 图尺寸

    Args:
        ref_image: REF 图像 (numpy array, BGR 或 RGB)
        src_size: SRC 图尺寸 (width, height)
        src_bbox: SRC 图中商品框 [x1, y1, x2, y2]
        ref_bbox: REF 图中商品框 [x1, y1, x2, y2]
        border_mode: 边界填充方式，可选：
            - "constant": 黑色填充（默认）
            - "reflect": 镜像填充
            - "replicate": 复制边缘像素
            - "background": 自动检测 REF 图背景色填充
        max_scale: 缩放倍率上限（如 2.0 表示最多放大 2 倍），0 表示不限制
        min_scale: 缩放倍率下限（如 0.5 表示最少缩小到 0.5 倍），0 表示不限制

    Returns:
        对齐后的 REF 图像，尺寸与 SRC 图一致
    """
    src_width, src_height = src_size
    ref_height, ref_width = ref_image.shape[:2]

    # 计算两个框的宽高
    src_box_w = src_bbox[2] - src_bbox[0]
    src_box_h = src_bbox[3] - src_bbox[1]
    ref_box_w = ref_bbox[2] - ref_bbox[0]
    ref_box_h = ref_bbox[3] - ref_bbox[1]

    # 计算等比缩放因子（几何均值，保证宽高比不变）
    scale_x = src_box_w / ref_box_w if ref_box_w > 0 else 1.0
    scale_y = src_box_h / ref_box_h if ref_box_h > 0 else 1.0
    uniform_scale = float(np.sqrt(scale_x * scale_y))

    # 应用 max_scale / min_scale 限制
    if max_scale > 0 and uniform_scale > max_scale:
        logger.info(f"无损对齐: 等比缩放因子 {uniform_scale:.3f} 超过上限 {max_scale:.2f}，已裁剪")
        uniform_scale = max_scale
    if min_scale > 0 and uniform_scale < min_scale:
        logger.info(f"无损对齐: 等比缩放因子 {uniform_scale:.3f} 低于下限 {min_scale:.2f}，已裁剪")
        uniform_scale = min_scale

    # 如果缩放因子接近 1.0，跳过缩放步骤
    skip_scaling = abs(uniform_scale - 1.0) < 0.01

    if skip_scaling:
        scaled_ref = ref_image
        scaled_ref_w, scaled_ref_h = ref_width, ref_height
        scaled_ref_center_x = (ref_bbox[0] + ref_bbox[2]) / 2.0
        scaled_ref_center_y = (ref_bbox[1] + ref_bbox[3]) / 2.0
        logger.info(f"无损对齐: 缩放因子 {uniform_scale:.3f} ≈ 1.0，跳过缩放")
    else:
        scaled_ref_w = int(round(ref_width * uniform_scale))
        scaled_ref_h = int(round(ref_height * uniform_scale))
        scaled_ref = cv2.resize(ref_image, (scaled_ref_w, scaled_ref_h),
                                interpolation=optimal_interpolation(uniform_scale))
        scaled_ref_center_x = (ref_bbox[0] + ref_bbox[2]) / 2.0 * uniform_scale
        scaled_ref_center_y = (ref_bbox[1] + ref_bbox[3]) / 2.0 * uniform_scale
        logger.info(f"无损对齐: 等比缩放 {uniform_scale:.3f} "
                    f"(REF {ref_width}x{ref_height} -> {scaled_ref_w}x{scaled_ref_h})")

    # 计算平移量：使缩放后的 REF 框中心对齐到 SRC 框中心
    src_center_x = (src_bbox[0] + src_bbox[2]) / 2.0
    src_center_y = (src_bbox[1] + src_bbox[3]) / 2.0
    shift_x = int(round(src_center_x - scaled_ref_center_x))
    shift_y = int(round(src_center_y - scaled_ref_center_y))

    logger.info(f"无损对齐: 平移量 dx={shift_x}, dy={shift_y}")

    # 检测填充色
    if border_mode == "background":
        fill_color = detect_background_color(ref_image)
        logger.info(f"检测到 REF 图背景色 (BGR): {fill_color}")
    else:
        fill_color = (0, 0, 0)

    # 计算源和目标的重叠区域
    channels = scaled_ref.shape[2] if len(scaled_ref.shape) == 3 else 1
    roi_src_x_start = max(0, -shift_x)
    roi_src_y_start = max(0, -shift_y)
    roi_src_x_end = min(scaled_ref_w, src_width - shift_x)
    roi_src_y_end = min(scaled_ref_h, src_height - shift_y)

    dst_x_start = max(0, shift_x)
    dst_y_start = max(0, shift_y)
    dst_x_end = dst_x_start + (roi_src_x_end - roi_src_x_start)
    dst_y_end = dst_y_start + (roi_src_y_end - roi_src_y_start)

    if roi_src_x_end <= roi_src_x_start or roi_src_y_end <= roi_src_y_start:
        logger.warning("无损对齐: 平移后无重叠区域，输出为纯填充图")
        return np.full((src_height, src_width, channels), fill_color, dtype=np.uint8)

    valid_region = scaled_ref[roi_src_y_start:roi_src_y_end, roi_src_x_start:roi_src_x_end]

    if border_mode in ("reflect", "replicate"):
        border_type = (cv2.BORDER_REFLECT_101 if border_mode == "reflect"
                       else cv2.BORDER_REPLICATE)
        canvas = cv2.copyMakeBorder(
            valid_region,
            top=dst_y_start,
            bottom=max(0, src_height - dst_y_end),
            left=dst_x_start,
            right=max(0, src_width - dst_x_end),
            borderType=border_type,
        )
        # copyMakeBorder 可能因 rounding 导致尺寸略有差异，裁剪到目标尺寸
        canvas = canvas[:src_height, :src_width]
    else:
        canvas = np.full((src_height, src_width, channels), fill_color, dtype=np.uint8)
        canvas[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = valid_region

    return canvas


def create_alignment_comparison(src_image_path: str,
                                ref_image_path: str,
                                aligned_ref_image: np.ndarray,
                                src_bbox: List[int],
                                ref_bbox: List[int],
                                output_path: str) -> str:
    """
    生成对齐前后的可视化对比图，方便检查对齐效果。

    输出一张横向拼接图：[SRC 原图 (带框)] | [REF 原图 (带框)] | [对齐后 REF (带框)]
    其中框使用 SRC 的商品框位置绘制，以验证对齐是否准确。

    Args:
        src_image_path: SRC 图像路径
        ref_image_path: REF 图像路径
        aligned_ref_image: 对齐后的 REF 图像 (numpy array, BGR)
        src_bbox: SRC 图中商品框 [x1, y1, x2, y2]
        ref_bbox: REF 图中商品框 [x1, y1, x2, y2]
        output_path: 对比图保存路径

    Returns:
        保存的输出路径
    """
    src_img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    ref_img = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
    src_height, src_width = src_img.shape[:2]

    # 在 SRC 图上画商品框（绿色）
    src_vis = src_img.copy()
    cv2.rectangle(src_vis, (src_bbox[0], src_bbox[1]), (src_bbox[2], src_bbox[3]),
                  (0, 255, 0), 2)
    cv2.putText(src_vis, "SRC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)

    # 将 REF 原图 resize 到与 SRC 同尺寸以便对比，并画原始框（蓝色）
    ref_resized = cv2.resize(ref_img, (src_width, src_height), interpolation=cv2.INTER_LANCZOS4)
    ref_h_orig, ref_w_orig = ref_img.shape[:2]
    scale_x = src_width / ref_w_orig
    scale_y = src_height / ref_h_orig
    scaled_ref_bbox = [
        int(ref_bbox[0] * scale_x), int(ref_bbox[1] * scale_y),
        int(ref_bbox[2] * scale_x), int(ref_bbox[3] * scale_y),
    ]
    cv2.rectangle(ref_resized,
                  (scaled_ref_bbox[0], scaled_ref_bbox[1]),
                  (scaled_ref_bbox[2], scaled_ref_bbox[3]),
                  (255, 0, 0), 2)
    cv2.putText(ref_resized, "REF (resized)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 0, 0), 2)

    # 在对齐后的 REF 上画 SRC 的商品框（绿色），验证对齐效果
    aligned_vis = aligned_ref_image.copy()
    cv2.rectangle(aligned_vis, (src_bbox[0], src_bbox[1]), (src_bbox[2], src_bbox[3]),
                  (0, 255, 0), 2)
    cv2.putText(aligned_vis, "Aligned REF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2)

    # 横向拼接，中间加 4px 白色分隔线
    separator = np.ones((src_height, 4, 3), dtype=np.uint8) * 255
    comparison = np.concatenate([src_vis, separator, ref_resized, separator, aligned_vis], axis=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, comparison)
    logger.info(f"对比图已保存到: {output_path}")
    return output_path

def read_txt_lines(txt_path: str) -> List[str]:
    """读取 txt 文件，返回非空行列表。"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def extract_product_name_from_prompt(prompt_txt_path: str) -> str:
    """
    从 prompt txt 文件中提取商品名称（用于 VLM grounding）。

    prompt 文件内容格式：
    - 裁剪图："{product_entity}：{product_description}"
    - 原图：  "{spatial_description} {product_entity}：{product_description}"

    提取 "：" 前面的最后一个词作为 product_entity，整行作为 product_name。
    """
    with open(prompt_txt_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    return content if content else "商品"


def paired_grounding(vlm_client: Client,
                     src_image_path: str,
                     ref_image_path: str,
                     product_name: str = "",
                     model: str = "qwen3.5-plus",
                     usage_id: str = "") -> Dict[str, Any]:
    """
    双图联合 grounding：一次输入 SRC + REF 两张图，
    让 VLM 同时返回两张图中对应同一商品区域的 bbox。

    Args:
        vlm_client: VLM Client 实例
        src_image_path: SRC 图像路径
        ref_image_path: REF 图像路径
        product_name: 商品名称/描述
        model: VLM 模型名称
        usage_id: 用量追踪 ID

    Returns:
        {"src_matched_bbox": [x1,y1,x2,y2], "ref_matched_bbox": [x1,y1,x2,y2], "match_description": "..."}
        bbox 为原图像素坐标。失败时 bbox 为 None。
    """
    canvas_width = 1080
    resize_version = "qwen3vl" if model.startswith("qwen3") else "qwen25vl"

    # 读取并缩放 SRC 图
    src_img = Image.open(src_image_path).convert("RGB")
    src_orig_w, src_orig_h = src_img.size
    src_canvas_h = int(canvas_width * src_orig_h / src_orig_w)
    if src_orig_w != canvas_width:
        src_img = src_img.resize((canvas_width, src_canvas_h), Image.Resampling.BICUBIC)
    src_base64 = pil_image_to_base64(src_img)

    # 读取并缩放 REF 图
    ref_img = Image.open(ref_image_path).convert("RGB")
    ref_orig_w, ref_orig_h = ref_img.size
    ref_canvas_h = int(canvas_width * ref_orig_h / ref_orig_w)
    if ref_orig_w != canvas_width:
        ref_img = ref_img.resize((canvas_width, ref_canvas_h), Image.Resampling.BICUBIC)
    ref_base64 = pil_image_to_base64(ref_img)

    # 构建 prompt
    prompt = PAIRED_GROUNDING_PROMPT
    if product_name:
        prompt += f"\n## 挂链商品信息\n{product_name}\n"

    image_urls = [src_base64, ref_base64]
    messages = [{
        "role": "user",
        "content": vlm_client.format_user_content(prompt, model=model, image_url=image_urls)
    }]

    response = vlm_client.chat(messages=messages, model=model, json_mode=False, usage_id=usage_id)
    result = parse_json(response)
    if result is None:
        result = vlm_client.convert_to_json(content=response, json_schema="合法的JSON对象", usage_id=usage_id)

    if not result:
        return {"src_matched_bbox": None, "ref_matched_bbox": None, "match_description": "VLM 解析失败"}

    # 将 src_matched_bbox 从 VLM 坐标系转换回 SRC 原图坐标系
    src_bbox_raw = result.get("src_matched_bbox")
    if src_bbox_raw and len(src_bbox_raw) == 4:
        mapped = resize_vlm_bboxes(
            [{"bbox_2d": src_bbox_raw}], src_canvas_h, canvas_width, version=resize_version
        )
        src_bbox_pixel = mapped[0]["bbox_2d"]
        # 从 canvas 坐标转回原图坐标
        result["src_matched_bbox"] = [
            int(src_bbox_pixel[0] * src_orig_w / canvas_width),
            int(src_bbox_pixel[1] * src_orig_h / src_canvas_h),
            int(src_bbox_pixel[2] * src_orig_w / canvas_width),
            int(src_bbox_pixel[3] * src_orig_h / src_canvas_h),
        ]

    # 将 ref_matched_bbox 从 VLM 坐标系转换回 REF 原图坐标系
    ref_bbox_raw = result.get("ref_matched_bbox")
    if ref_bbox_raw and len(ref_bbox_raw) == 4:
        mapped = resize_vlm_bboxes(
            [{"bbox_2d": ref_bbox_raw}], ref_canvas_h, canvas_width, version=resize_version
        )
        ref_bbox_pixel = mapped[0]["bbox_2d"]
        # 从 canvas 坐标转回原图坐标
        result["ref_matched_bbox"] = [
            int(ref_bbox_pixel[0] * ref_orig_w / canvas_width),
            int(ref_bbox_pixel[1] * ref_orig_h / ref_canvas_h),
            int(ref_bbox_pixel[2] * ref_orig_w / canvas_width),
            int(ref_bbox_pixel[3] * ref_orig_h / ref_canvas_h),
        ]

    return result


def main():
    """
    从测试集的 txt 文件列表输入，调用 VLM 双图联合 grounding 后做 REF→HQ 对齐。

    输入：hq_ori.txt、ref_ori.txt、prompt_crop.txt（逐行对应）
    输出：测试集目录下 img/ref_aligned/ 和 ref_aligned.txt
    """
    parser = argparse.ArgumentParser(
        description="从测试集 txt 文件列表输入，VLM grounding + REF→HQ 对齐"
    )
    parser.add_argument("--hq-txt", type=str, required=True,
                        help="HQ 图像路径列表文件（如 hq_ori.txt）")
    parser.add_argument("--ref-txt", type=str, required=True,
                        help="REF 图像路径列表文件（如 ref_ori.txt）")
    parser.add_argument("--prompt-txt", type=str, required=True,
                        help="Prompt 文件路径列表文件（如 prompt_crop.txt），每行一个 .txt 路径")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="测试集根目录（输出到其下的 img/ref_aligned/ 和 ref_aligned.txt）")
    parser.add_argument("--border-mode", type=str, default="constant",
                        choices=["reflect", "constant", "replicate", "background"],
                        help="边界填充方式: reflect/constant/replicate/background (默认: constant)")
    parser.add_argument("--max-aspect-ratio-change", type=float, default=1.3,
                        help="限制商品宽高比变化上限，0 表示不限制（默认: 1.3）")
    parser.add_argument("--max-scale", type=float, default=2.0,
                        help="缩放倍率上限（如 2.0 表示最多放大 2 倍），0 表示不限制（默认: 2.0）")
    parser.add_argument("--min-scale", type=float, default=0.5,
                        help="缩放倍率下限（如 0.5 表示最少缩小到 0.5 倍），0 表示不限制（默认: 0.5）")
    parser.add_argument("--save-comparison", action="store_true",
                        help="是否同时保存对齐前后的可视化对比图")
    parser.add_argument("--model", type=str, default="qwen3.5-plus",
                        help="VLM grounding 模型名称（默认: qwen3.5-plus）")
    parser.add_argument("--api-provider", type=str, default="dashscope",
                        help="VLM API 提供方（默认: dashscope）")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="VLM grounding 并发线程数（默认: 4）")
    parser.add_argument("--lossless", action="store_true",
                        help="宽松无损对齐模式：仅等比缩放+平移+padding，不改变商品宽高比")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 读取三个 txt 文件
    hq_paths = read_txt_lines(args.hq_txt)
    ref_paths = read_txt_lines(args.ref_txt)
    prompt_paths = read_txt_lines(args.prompt_txt)

    if not (len(hq_paths) == len(ref_paths) == len(prompt_paths)):
        print(f"ERROR: 行数不一致！hq={len(hq_paths)}, ref={len(ref_paths)}, prompt={len(prompt_paths)}")
        return

    print("=" * 60)
    print("REF → HQ 对齐（从 txt 文件列表 + VLM Grounding）")
    print("=" * 60)
    print(f"  HQ 列表:     {args.hq_txt} ({len(hq_paths)} 条)")
    print(f"  REF 列表:    {args.ref_txt} ({len(ref_paths)} 条)")
    print(f"  Prompt 列表: {args.prompt_txt} ({len(prompt_paths)} 条)")
    print(f"  输出目录:    {args.output_dir}")
    print(f"  边界填充:    {args.border_mode}")
    print(f"  宽高比限制:  {args.max_aspect_ratio_change if args.max_aspect_ratio_change > 0 else '不限制'}")
    print(f"  最大缩放:    {args.max_scale if args.max_scale > 0 else '不限制'}")
    print(f"  最小缩放:    {args.min_scale if args.min_scale > 0 else '不限制'}")
    print(f"  VLM 模型:    {args.model}")
    print(f"  并发线程数:  {args.num_workers}")
    print(f"  无损模式:    {args.lossless}")
    print(f"  保存对比图:  {args.save_comparison}")
    print("=" * 60)

    # 初始化 VLM
    vlm_client = Client(api_provider=args.api_provider)

    # 创建输出目录
    aligned_ref_dir = os.path.join(args.output_dir, "img", "ref_aligned")
    os.makedirs(aligned_ref_dir, exist_ok=True)
    if args.save_comparison:
        comparison_dir = os.path.join(args.output_dir, "img", "alignment_comparison")
        os.makedirs(comparison_dir, exist_ok=True)

    from tqdm import tqdm

    # ---- 阶段 1：预检查文件存在性，构建有效任务列表 ----
    valid_tasks = []
    total_skipped = 0

    for idx, (hq_path, ref_path, prompt_path) in enumerate(
            zip(hq_paths, ref_paths, prompt_paths)):
        sample_id = f"{idx:04d}"

        if not os.path.exists(hq_path):
            print(f"  [{sample_id}] Skipping: HQ image not found: {hq_path}")
            total_skipped += 1
            continue
        if not os.path.exists(ref_path):
            print(f"  [{sample_id}] Skipping: REF image not found: {ref_path}")
            total_skipped += 1
            continue
        if not os.path.exists(prompt_path):
            print(f"  [{sample_id}] Skipping: Prompt file not found: {prompt_path}")
            total_skipped += 1
            continue

        product_name = extract_product_name_from_prompt(prompt_path)
        valid_tasks.append({
            "idx": idx,
            "sample_id": sample_id,
            "hq_path": hq_path,
            "ref_path": ref_path,
            "product_name": product_name,
        })

    # ---- 阶段 2：并发执行 paired_grounding ----
    def run_grounding(task: Dict[str, Any]) -> Dict[str, Any]:
        """在线程中执行单个 paired_grounding 任务。"""
        sample_id = task["sample_id"]
        try:
            vlm_client.get_usages(usage_id=sample_id, reset=True)
            grounding_result = paired_grounding(
                vlm_client=vlm_client,
                src_image_path=task["hq_path"],
                ref_image_path=task["ref_path"],
                product_name=task["product_name"],
                model=args.model,
                usage_id=sample_id,
            )
            vlm_client.get_usages(usage_id=sample_id, reset=False)
            return {**task, "grounding_result": grounding_result, "error": None}
        except Exception as exc:
            return {**task, "grounding_result": None, "error": str(exc)}

    grounding_results = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {executor.submit(run_grounding, task): task for task in valid_tasks}

        with tqdm(total=len(valid_tasks), desc="VLM Grounding") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                grounding_results[result["idx"]] = result
                status = "✓" if result["error"] is None else "✗"
                pbar.set_postfix({"sample": result["sample_id"], "status": status})
                pbar.update(1)

    # ---- 阶段 3：按原始顺序执行对齐和保存 ----
    # grounding 失败时 fallback 为直接 resize REF 图到 HQ 尺寸，保证输出与输入一一对应
    aligned_ref_output_paths = []
    all_stats = []
    total_aligned = 0
    total_fallback = 0

    for task in tqdm(valid_tasks, desc="Aligning"):
        idx = task["idx"]
        sample_id = task["sample_id"]
        hq_path = task["hq_path"]
        ref_path = task["ref_path"]

        grounding_data = grounding_results[idx]

        # 获取 SRC 图像尺寸
        src_img = Image.open(hq_path).convert("RGB")
        src_width, src_height = src_img.size

        # 读取 REF 图像
        ref_img_cv = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        if ref_img_cv is None:
            print(f"  [{sample_id}] WARNING: failed to read REF image, using blank image")
            ref_img_cv = np.zeros((src_height, src_width, 3), dtype=np.uint8)

        # 判断 grounding 是否成功
        use_fallback = False
        src_bbox = None
        ref_bbox = None
        fallback_reason = ""

        if grounding_data["error"] is not None:
            use_fallback = True
            fallback_reason = f"grounding failed: {grounding_data['error']}"
        else:
            grounding_result = grounding_data["grounding_result"]
            src_bbox = grounding_result.get("src_matched_bbox")
            ref_bbox = grounding_result.get("ref_matched_bbox")

            if not src_bbox or len(src_bbox) != 4:
                use_fallback = True
                fallback_reason = "invalid SRC bbox"
            elif not ref_bbox or len(ref_bbox) != 4:
                use_fallback = True
                fallback_reason = "invalid REF bbox"

        if use_fallback:
            # Fallback：直接将 REF 图 resize 到 SRC 尺寸
            print(f"  [{sample_id}] Fallback (resize REF): {fallback_reason}")
            ref_h, ref_w = ref_img_cv.shape[:2]
            fallback_scale = float(np.sqrt(
                (src_width / ref_w) * (src_height / ref_h)
            )) if ref_w > 0 and ref_h > 0 else 1.0
            aligned = cv2.resize(ref_img_cv, (src_width, src_height),
                                 interpolation=optimal_interpolation(fallback_scale))
            total_fallback += 1
        else:
            match_description = grounding_result.get("match_description", "")
            if match_description:
                logger.info(f"  [{sample_id}] Matched: {match_description}")

            if args.lossless:
                # 无损对齐：等比缩放 + 平移 + padding
                aligned = align_ref_to_hq_lossless(
                    ref_image=ref_img_cv,
                    src_size=(src_width, src_height),
                    src_bbox=src_bbox,
                    ref_bbox=ref_bbox,
                    border_mode=args.border_mode,
                    max_scale=args.max_scale,
                    min_scale=args.min_scale,
                )
            else:
                # 标准仿射对齐
                aligned = align_ref_to_hq(
                    ref_image=ref_img_cv,
                    src_size=(src_width, src_height),
                    src_bbox=src_bbox,
                    ref_bbox=ref_bbox,
                    border_mode=args.border_mode,
                    max_aspect_ratio_change=args.max_aspect_ratio_change,
                    max_scale=args.max_scale,
                    min_scale=args.min_scale,
                )
            total_aligned += 1

        # 保存对齐后的 REF 图像（文件名与 HQ 一致）
        src_basename = os.path.splitext(os.path.basename(hq_path))[0]
        output_filename = f"{src_basename}_ref_aligned.png"
        output_path = os.path.join(aligned_ref_dir, output_filename)
        cv2.imwrite(output_path, aligned)
        aligned_ref_output_paths.append(output_path)

        stat = {
            "index": idx,
            "hq_image": hq_path,
            "ref_image": ref_path,
            "src_bbox": src_bbox,
            "ref_bbox": ref_bbox,
            "src_size": [src_width, src_height],
            "fallback": use_fallback,
            "fallback_reason": fallback_reason if use_fallback else "",
            "aligned_ref_path": output_path,
        }

        if not use_fallback:
            src_box_w = src_bbox[2] - src_bbox[0]
            src_box_h = src_bbox[3] - src_bbox[1]
            ref_box_w = ref_bbox[2] - ref_bbox[0]
            ref_box_h = ref_bbox[3] - ref_bbox[1]
            stat["scale_x"] = src_box_w / ref_box_w if ref_box_w > 0 else 0
            stat["scale_y"] = src_box_h / ref_box_h if ref_box_h > 0 else 0
            if args.lossless:
                stat["uniform_scale"] = float(np.sqrt(stat["scale_x"] * stat["scale_y"])) if stat["scale_x"] > 0 and stat["scale_y"] > 0 else 1.0
                stat["align_mode"] = "lossless"
            else:
                stat["align_mode"] = "affine"

        # 可选：保存对比图（仅对齐成功时有意义）
        if args.save_comparison and not use_fallback:
            comp_filename = f"{src_basename}_alignment_comparison.jpg"
            comp_path = os.path.join(comparison_dir, comp_filename)
            create_alignment_comparison(
                src_image_path=hq_path,
                ref_image_path=ref_path,
                aligned_ref_image=aligned,
                src_bbox=src_bbox,
                ref_bbox=ref_bbox,
                output_path=comp_path,
            )
            stat["comparison_path"] = comp_path

        all_stats.append(stat)

    # 写入 ref_aligned.txt
    ref_aligned_txt = os.path.join(args.output_dir, "ref_aligned.txt")
    write_image_paths_to_file(ref_aligned_txt, aligned_ref_output_paths)

    # 保存统计信息
    stats_path = os.path.join(args.output_dir, "alignment_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("REF → HQ 对齐完成（VLM Grounding 模式）")
    print("=" * 60)
    print(f"  总样本数:     {len(hq_paths)}")
    print(f"  文件缺失跳过: {total_skipped}")
    print(f"  成功对齐:     {total_aligned}")
    print(f"  Fallback:     {total_fallback}")
    print(f"  输出总数:     {len(aligned_ref_output_paths)}")
    print(f"  对齐图像目录: {aligned_ref_dir}")
    print(f"  路径列表:     {ref_aligned_txt}")
    print(f"  统计信息:     {stats_path}")
    if args.save_comparison:
        print(f"  对比图目录:   {comparison_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
