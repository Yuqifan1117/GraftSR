#!/usr/bin/env python3
"""
从 CSV 文件直接生成真实世界测试集。

输出目录结构：
  output_dir/
    img/
      lq/              - 下载的 LQ 图像
      ref/             - 下载的 REF 图像
      lq_crop/         - LQ 图像商品框裁剪
      ref_crop/        - REF 图像商品框裁剪
      ref_aligned/     - 对齐后的 REF 图像（与 LQ 尺寸一致，商品框对齐）
      alignment_comparison/  - 对齐前后可视化对比图（可选）
      tile_ref_texture_ori/      - per-tile 纹理块（原图裁剪）
      tile_ref_texture_resized/  - per-tile 纹理块（缩小后裁剪）
      tile_texture_maps/         - per-tile 纹理映射 JSON
    prompt/
      lq/              - LQ 图像对应的 prompt 文件（旧版：paired_grounding 拼接）
      lq_opt/           - LQ 图像对应的高质量 prompt 文件（新版：VLM 独立标注 SRC + REF）
    lq.txt             - LQ 图像路径索引
    ref.txt            - REF 图像路径索引
    lq_crop.txt        - LQ 商品框裁剪路径索引
    ref_crop.txt       - REF 商品框裁剪路径索引
    ref_aligned.txt    - 对齐后 REF 图像路径索引
    prompt.txt         - prompt 文件路径索引（旧版）
    prompt_opt.txt     - prompt 文件路径索引（新版高质量）
    ...
    generation_stats.json  - 生成统计信息
"""
import argparse
import csv
import json
import logging
import os
from typing import Dict, Any, List, Optional
from tqdm import tqdm

import cv2
from PIL import Image

from image_utils import download_image, crop_image
from convert_utils import write_image_paths_to_file
from client import Client
from align_ref_to_hq import (paired_grounding, align_ref_to_hq,
                              align_ref_to_hq_lossless,
                              create_alignment_comparison)
from prompts import IMAGE_CAPTION_PROMPT, PRODUCT_CAPTION_PROMPT


logger = logging.getLogger(__name__)


def caption_src_image(vlm_client: Client, image_path: str, model: str, usage_id: str = "") -> str:
    """使用 IMAGE_CAPTION_PROMPT 对 SRC 图像做整体视觉内容描述。"""
    try:
        messages = [{
            "role": "user",
            "content": vlm_client.format_user_content(
                IMAGE_CAPTION_PROMPT, model=model, image_url=image_path,
            )
        }]
        return vlm_client.chat(
            messages=messages, model=model, json_mode=False, usage_id=usage_id,
        ).strip()
    except Exception as e:
        logger.error(f"Error captioning SRC image {image_path}: {e}")
        return ""


def caption_ref_product(vlm_client: Client, image_path: str, model: str, usage_id: str = "") -> str:
    """使用 PRODUCT_CAPTION_PROMPT 对 REF 商品裁剪图做商品视觉特征描述。"""
    try:
        messages = [{
            "role": "user",
            "content": vlm_client.format_user_content(
                PRODUCT_CAPTION_PROMPT, model=model, image_url=image_path,
            )
        }]
        return vlm_client.chat(
            messages=messages, model=model, json_mode=False, usage_id=usage_id,
        ).strip()
    except Exception as e:
        logger.error(f"Error captioning REF product {image_path}: {e}")
        return ""


def resize_image_by_short_side(image_path: str, max_short_side: int) -> bool:
    """
    如果图像短边超过 max_short_side，则等比缩放并覆盖原文件。

    Args:
        image_path: 图像文件路径
        max_short_side: 短边最大像素值

    Returns:
        True 表示进行了缩放，False 表示无需缩放
    """
    img = Image.open(image_path)
    width, height = img.size
    short_side = min(width, height)
    if short_side <= max_short_side:
        return False
    scale = max_short_side / short_side
    new_width = int(width * scale)
    new_height = int(height * scale)
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    img_resized.save(image_path)
    logger.info(f"Resized {image_path}: ({width}x{height}) -> ({new_width}x{new_height})")
    return True


def read_csv_and_filter(csv_path: str) -> List[Dict[str, str]]:
    """
    读取 CSV 文件并过滤出"采用"列为"是"的行。

    Args:
        csv_path: CSV 文件路径

    Returns:
        过滤后的行列表，每行是一个字典
    """
    filtered_rows = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            adopted = row.get("二次筛选", row.get("采用", "")).strip()
            if adopted == "是":
                filtered_rows.append({
                    "item_id": row["item_id"].strip(),
                    "item_title": row["item_title"].strip(),
                    "lq_image_url": row["lq_image_url"].strip().strip('"'),
                    "ref_image_url": row["ref_image_url"].strip().strip('"'),
                })
    return filtered_rows


def get_image_extension_from_url(url: str) -> str:
    """从 URL 中提取图像扩展名，默认返回 .jpg"""
    path = url.split("?")[0]
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        return ext
    return ".jpg"


def create_prompt_file_direct(prompt_path: str, prompt_text: str) -> str:
    """
    直接创建 prompt 文件（不依赖 convert_utils 的路径替换逻辑）。

    Args:
        prompt_path: prompt 文件保存路径
        prompt_text: prompt 文本内容

    Returns:
        prompt 文件路径，内容为空时返回 None
    """
    if not prompt_text:
        return None
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt_text)
    return prompt_path


def process_single_item(row: Dict[str, str],
                        output_img_dir: str,
                        output_prompt_dir: str,
                        vlm_client: Client = None,
                        grounding_model: str = "qwen3.5-plus",
                        caption_model: str = "qwen3.5-plus",
                        border_mode: str = "background",
                        max_aspect_ratio_change: float = 1.3,
                        max_scale: float = 2.0,
                        min_scale: float = 0.5,
                        lossless: bool = False,
                        save_comparison: bool = False,
                        max_short_side: int = 0) -> Optional[Dict[str, Any]]:
    """
    处理单条 CSV 记录：
    1. 下载 LQ 和 REF 图像
    2. 通过 paired_grounding 获取双图商品框 + 商品描述
    3. 基于商品框做 REF→LQ 对齐
    4. 用 VLM 返回的 product_entity + product_description 生成 prompt
    5. 可选生成 per-tile 纹理映射

    Args:
        row: CSV 行数据
        output_img_dir: 图像输出根目录
        output_prompt_dir: prompt 输出根目录
        vlm_client: VLM Client 实例
        grounding_model: paired grounding VLM 模型名称
        border_mode: REF 对齐时的边界填充方式
        max_aspect_ratio_change: REF 对齐时的宽高比变化上限
        max_scale: 缩放倍率上限，0 表示不限制
        min_scale: 缩放倍率下限，0 表示不限制
        save_comparison: 是否保存对齐前后的可视化对比图
        enable_tile_texture: 是否启用 per-tile 纹理匹配
        tile_texture_model: tile 纹理匹配 VLM 模型名称

    Returns:
        处理结果字典，失败返回 None
    """
    item_id = row["item_id"]
    item_title = row["item_title"]
    lq_url = row["lq_image_url"]
    ref_url = row["ref_image_url"]

    # 确定文件名
    lq_ext = get_image_extension_from_url(lq_url)
    ref_ext = get_image_extension_from_url(ref_url)
    lq_filename = f"{item_id}_lq{lq_ext}"
    ref_filename = f"{item_id}_ref{ref_ext}"

    cdn_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    # 下载 LQ 图像
    lq_dir = os.path.join(output_img_dir, "lq")
    os.makedirs(lq_dir, exist_ok=True)
    lq_save_path = os.path.join(lq_dir, lq_filename)
    lq_path = download_image(lq_url, lq_save_path, headers=cdn_headers)
    if not lq_path:
        print(f"  [{item_id}] SKIP: Failed to download LQ image: {lq_url}")
        return None

    # 下载 REF 图像
    ref_dir = os.path.join(output_img_dir, "ref")
    os.makedirs(ref_dir, exist_ok=True)
    ref_save_path = os.path.join(ref_dir, ref_filename)
    ref_path = download_image(ref_url, ref_save_path, headers=cdn_headers)
    if not ref_path:
        print(f"  [{item_id}] SKIP: Failed to download REF image: {ref_url}")
        if os.path.exists(lq_save_path):
            os.remove(lq_save_path)
        return None

    # 短边限制缩放
    if max_short_side > 0:
        resize_image_by_short_side(lq_path, max_short_side)
        resize_image_by_short_side(ref_path, max_short_side)

    # ---- Paired Grounding：获取双图商品框 + 商品描述 ----
    vlm_client.get_usages(usage_id=item_id, reset=True)

    try:
        grounding_result = paired_grounding(
            vlm_client=vlm_client,
            src_image_path=lq_path,
            ref_image_path=ref_path,
            product_name=item_title,
            model=grounding_model,
            usage_id=item_id,
        )
    except Exception as e:
        print(f"  [{item_id}] SKIP: paired grounding failed: {e}")
        return None

    lq_bbox = grounding_result.get("src_matched_bbox")
    ref_bbox = grounding_result.get("ref_matched_bbox")

    if not lq_bbox or len(lq_bbox) != 4 or not ref_bbox or len(ref_bbox) != 4:
        print(f"  [{item_id}] SKIP: invalid bbox from paired grounding "
              f"(lq_bbox={lq_bbox}, ref_bbox={ref_bbox})")
        return None

    # 提取 VLM 返回的商品描述字段
    product_entity = grounding_result.get("product_entity", "")
    product_description = grounding_result.get("product_description", "")
    lq_image_description = grounding_result.get("src_image_description", "")
    lq_image_ocr = grounding_result.get("src_image_ocr", "")
    match_description = grounding_result.get("match_description", "")

    if match_description:
        logger.info(f"  [{item_id}] Matched: {match_description}")

    # ---- REF→LQ 对齐 ----
    lq_img = Image.open(lq_path).convert("RGB")
    lq_width, lq_height = lq_img.size

    ref_img_cv = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_img_cv is None:
        print(f"  [{item_id}] SKIP: failed to read REF image for alignment")
        return None

    if lossless:
        aligned_ref = align_ref_to_hq_lossless(
            ref_image=ref_img_cv,
            src_size=(lq_width, lq_height),
            src_bbox=lq_bbox,
            ref_bbox=ref_bbox,
            border_mode=border_mode,
            max_scale=max_scale,
            min_scale=min_scale,
        )
    else:
        aligned_ref = align_ref_to_hq(
            ref_image=ref_img_cv,
            src_size=(lq_width, lq_height),
            src_bbox=lq_bbox,
            ref_bbox=ref_bbox,
            border_mode=border_mode,
            max_aspect_ratio_change=max_aspect_ratio_change,
            max_scale=max_scale,
            min_scale=min_scale,
        )

    # ---- 保存 LQ crop 和 REF crop（根据 bbox 裁剪商品区域）----
    lq_crop_dir = os.path.join(output_img_dir, "lq_crop")
    os.makedirs(lq_crop_dir, exist_ok=True)
    lq_crop_filename = f"{item_id}_lq_crop.png"
    lq_crop_path = os.path.join(lq_crop_dir, lq_crop_filename)
    crop_image(lq_bbox, lq_path, lq_crop_path, lossless=True)

    ref_crop_dir = os.path.join(output_img_dir, "ref_crop")
    os.makedirs(ref_crop_dir, exist_ok=True)
    ref_crop_filename = f"{item_id}_ref_crop.png"
    ref_crop_path = os.path.join(ref_crop_dir, ref_crop_filename)
    # 将 ref_bbox 上下左右各扩充 10%
    ref_x1, ref_y1, ref_x2, ref_y2 = ref_bbox
    ref_bbox_w = ref_x2 - ref_x1
    ref_bbox_h = ref_y2 - ref_y1
    expand_x = int(ref_bbox_w * 0.1)
    expand_y = int(ref_bbox_h * 0.1)
    ref_bbox_expanded = [
        ref_x1 - expand_x,
        ref_y1 - expand_y,
        ref_x2 + expand_x,
        ref_y2 + expand_y,
    ]
    crop_image(ref_bbox_expanded, ref_path, ref_crop_path, lossless=True)

    # ---- VLM 独立标注 SRC 和 REF 商品裁剪图 ----
    src_caption = caption_src_image(
        vlm_client, lq_path, model=caption_model, usage_id=item_id
    )
    ref_product_caption = caption_ref_product(
        vlm_client, ref_crop_path, model=caption_model, usage_id=item_id
    )
    logger.info(f"  [{item_id}] SRC caption: {src_caption[:80]}...")
    logger.info(f"  [{item_id}] REF product caption: {ref_product_caption[:80]}...")

    # 保存对齐后的 REF 图像
    ref_aligned_dir = os.path.join(output_img_dir, "ref_aligned")
    os.makedirs(ref_aligned_dir, exist_ok=True)
    aligned_filename = f"{item_id}_ref_aligned.png"
    aligned_ref_path = os.path.join(ref_aligned_dir, aligned_filename)
    cv2.imwrite(aligned_ref_path, aligned_ref)

    # 可选：保存对比图
    comparison_path = None
    if save_comparison:
        comparison_dir = os.path.join(output_img_dir, "alignment_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        comp_filename = f"{item_id}_alignment_comparison.jpg"
        comparison_path = os.path.join(comparison_dir, comp_filename)
        create_alignment_comparison(
            src_image_path=lq_path,
            ref_image_path=ref_path,
            aligned_ref_image=aligned_ref,
            src_bbox=lq_bbox,
            ref_bbox=ref_bbox,
            output_path=comparison_path,
        )

    # ---- 生成 prompt 文件（使用 VLM 返回的 product_entity + product_description）----
    prompt_dir = os.path.join(output_prompt_dir, "lq")
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_filename = f"{item_id}_lq.txt"
    prompt_save_path = os.path.join(prompt_dir, prompt_filename)

    # prompt 格式与训练集一致："{lq_image_description} {lq_image_ocr} {product_entity}：{product_description}"
    if product_entity and product_description and lq_image_description and lq_image_ocr:
        prompt_text = f"{lq_image_description} {lq_image_ocr} {product_entity}：{product_description}"
    elif product_entity and product_description and lq_image_description:
        prompt_text = f"{lq_image_description} {product_entity}：{product_description}"
    elif product_entity and product_description:
        prompt_text = f"{product_entity}：{product_description}"
    elif product_entity:
        prompt_text = product_entity
    else:
        prompt_text = item_title
    prompt_path = create_prompt_file_direct(prompt_save_path, prompt_text)

    # ---- 生成新版高质量 prompt 文件（VLM 独立标注 SRC + REF）----
    prompt_opt_dir = os.path.join(output_prompt_dir, "lq_opt")
    os.makedirs(prompt_opt_dir, exist_ok=True)
    prompt_opt_filename = f"{item_id}_lq.txt"
    prompt_opt_save_path = os.path.join(prompt_opt_dir, prompt_opt_filename)

    prompt_opt_parts = []
    if src_caption:
        prompt_opt_parts.append(src_caption)
    if lq_image_ocr:
        prompt_opt_parts.append(lq_image_ocr)
    if product_entity and ref_product_caption:
        prompt_opt_parts.append(f"{product_entity}：{ref_product_caption}")
    elif product_entity and product_description:
        prompt_opt_parts.append(f"{product_entity}：{product_description}")
    elif product_entity:
        prompt_opt_parts.append(product_entity)

    prompt_opt_text = " ".join(prompt_opt_parts) if prompt_opt_parts else item_title
    prompt_opt_path = create_prompt_file_direct(prompt_opt_save_path, prompt_opt_text)

    sample = {
        "item_id": item_id,
        "item_title": item_title,
        "lq": lq_path,
        "ref": ref_path,
        "lq_crop": lq_crop_path,
        "ref_crop": ref_crop_path,
        "ref_aligned": aligned_ref_path,
        "prompt": prompt_path,
        "prompt_opt": prompt_opt_path,
        "lq_bbox": lq_bbox,
        "ref_bbox": ref_bbox,
        "product_entity": product_entity,
        "product_description": product_description,
        "lq_image_description": lq_image_description,
        "lq_image_ocr": lq_image_ocr,
        "match_description": match_description,
        "src_caption": src_caption,
        "ref_product_caption": ref_product_caption,
        "comparison": comparison_path,
    }

    vlm_client.get_usages(usage_id=item_id)
    return sample


def generate_realworld_testset(csv_path: str,
                               output_dir: str,
                               grounding_model: str = "qwen3.5-plus",
                               caption_model: str = "qwen3.5-plus",
                               border_mode: str = "background",
                               max_aspect_ratio_change: float = 1.3,
                               max_scale: float = 2.0,
                               min_scale: float = 0.5,
                               lossless: bool = False,
                               save_comparison: bool = False,
                               max_short_side: int = 0) -> Dict[str, Any]:
    """
    从 CSV 文件生成真实世界测试集。

    Args:
        csv_path: CSV 文件路径
        output_dir: 输出目录
        grounding_model: paired grounding VLM 模型名称
        border_mode: REF 对齐时的边界填充方式
        max_aspect_ratio_change: REF 对齐时的宽高比变化上限
        max_scale: 缩放倍率上限，0 表示不限制
        min_scale: 缩放倍率下限，0 表示不限制
        lossless: 是否使用宽松无损对齐模式（等比缩放+平移+padding）
        save_comparison: 是否保存对齐前后的可视化对比图

    Returns:
        生成统计信息
    """
    output_img_dir = os.path.join(output_dir, "img")
    output_prompt_dir = os.path.join(output_dir, "prompt")
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_prompt_dir, exist_ok=True)

    # 读取并过滤 CSV
    rows = read_csv_and_filter(csv_path)
    print(f"CSV total rows: read and filtered to {len(rows)} adopted items")

    if not rows:
        print("No adopted items found in CSV. Exiting.")
        return {"total_csv_rows": 0, "total_samples": 0}

    # 初始化 VLM client
    vlm_client = Client(api_provider="dashscope")
    print(f"Grounding model: {grounding_model}")

    # 收集路径
    lq_paths = []
    ref_paths = []
    lq_crop_paths = []
    ref_crop_paths = []
    ref_aligned_paths = []
    prompt_paths = []
    prompt_opt_paths = []
    all_samples = []
    failed_count = 0

    for row in tqdm(rows, desc="Generating realworld testset"):
        try:
            sample = process_single_item(
                row, output_img_dir, output_prompt_dir,
                vlm_client=vlm_client,
                grounding_model=grounding_model,
                caption_model=caption_model,
                border_mode=border_mode,
                max_aspect_ratio_change=max_aspect_ratio_change,
                max_scale=max_scale,
                min_scale=min_scale,
                lossless=lossless,
                save_comparison=save_comparison,
                max_short_side=max_short_side,
            )
            if sample is None:
                failed_count += 1
                continue

            all_samples.append(sample)
            lq_paths.append(sample["lq"])
            ref_paths.append(sample["ref"])
            lq_crop_paths.append(sample["lq_crop"])
            ref_crop_paths.append(sample["ref_crop"])
            ref_aligned_paths.append(sample["ref_aligned"])
            if sample.get("prompt"):
                prompt_paths.append(sample["prompt"])
            if sample.get("prompt_opt"):
                prompt_opt_paths.append(sample["prompt_opt"])

        except Exception as e:
            print(f"Error processing {row.get('item_id', 'unknown')}: {e}")
            failed_count += 1
            continue

    # 写入索引文件
    lq_txt = os.path.join(output_dir, "lq.txt")
    ref_txt = os.path.join(output_dir, "ref.txt")
    lq_crop_txt = os.path.join(output_dir, "lq_crop.txt")
    ref_crop_txt = os.path.join(output_dir, "ref_crop.txt")
    ref_aligned_txt = os.path.join(output_dir, "ref_aligned.txt")
    prompt_txt = os.path.join(output_dir, "prompt.txt")
    prompt_opt_txt = os.path.join(output_dir, "prompt_opt.txt")

    write_image_paths_to_file(lq_txt, lq_paths)
    write_image_paths_to_file(ref_txt, ref_paths)
    write_image_paths_to_file(lq_crop_txt, lq_crop_paths)
    write_image_paths_to_file(ref_crop_txt, ref_crop_paths)
    write_image_paths_to_file(ref_aligned_txt, ref_aligned_paths)
    write_image_paths_to_file(prompt_txt, prompt_paths)
    write_image_paths_to_file(prompt_opt_txt, prompt_opt_paths)

    # 统计信息
    stats = {
        "csv_path": csv_path,
        "total_adopted_rows": len(rows),
        "total_samples": len(all_samples),
        "failed_count": failed_count,
        "total_lq": len(lq_paths),
        "total_ref": len(ref_paths),
        "total_lq_crop": len(lq_crop_paths),
        "total_ref_crop": len(ref_crop_paths),
        "total_ref_aligned": len(ref_aligned_paths),
        "total_prompt": len(prompt_paths),
        "total_prompt_opt": len(prompt_opt_paths),
        "output_dir": output_dir,
        "grounding_model": grounding_model,
        "caption_model": caption_model,
        "border_mode": border_mode,
        "max_aspect_ratio_change": max_aspect_ratio_change,
        "dataset_files": {
            "lq_txt": lq_txt,
            "ref_txt": ref_txt,
            "lq_crop_txt": lq_crop_txt,
            "ref_crop_txt": ref_crop_txt,
            "ref_aligned_txt": ref_aligned_txt,
            "prompt_txt": prompt_txt,
            "prompt_opt_txt": prompt_opt_txt,
        },
        "samples": all_samples,
    }

    # 保存统计信息
    stats_json_path = os.path.join(output_dir, "generation_stats.json")
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate realworld test dataset from CSV")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to CSV file (columns: item_id, item_title, lq_image_url, ref_image_url, 采用)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for test data")
    parser.add_argument("--grounding-model", type=str, default="qwen3.5-plus",
                        help="VLM model for paired grounding (default: qwen3.5-plus)")
    parser.add_argument("--border-mode", type=str, default="constant",
                        choices=["reflect", "constant", "replicate", "background"],
                        help="REF alignment border mode (default: background)")
    parser.add_argument("--max-aspect-ratio-change", type=float, default=1.3,
                        help="Max aspect ratio change for REF alignment (default: 1.3)")
    parser.add_argument("--max-scale", type=float, default=2.0,
                        help="Max scale factor for REF alignment, 0 means no limit (default: 2.0)")
    parser.add_argument("--min-scale", type=float, default=0.5,
                        help="Min scale factor for REF alignment, 0 means no limit (default: 0.5)")
    parser.add_argument("--save-comparison", action="store_true",
                        help="Save alignment comparison images for visual inspection")
    parser.add_argument("--lossless", action="store_true",
                        help="Lossless alignment mode: uniform scaling + translation + padding only")
    parser.add_argument("--max-short-side", type=int, default=0,
                        help="Max short side of downloaded images in pixels, 0 means no limit (default: 2048)")
    parser.add_argument("--caption-model", type=str, default="qwen3.5-plus",
                        help="VLM model for image/product captioning (default: qwen3.5-plus)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    stats = generate_realworld_testset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        grounding_model=args.grounding_model,
        caption_model=args.caption_model,
        border_mode=args.border_mode,
        max_aspect_ratio_change=args.max_aspect_ratio_change,
        max_scale=args.max_scale,
        min_scale=args.min_scale,
        lossless=args.lossless,
        save_comparison=args.save_comparison,
        max_short_side=args.max_short_side,
    )

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Realworld test dataset generation completed!")
    print("=" * 60)
    print(f"CSV adopted rows: {stats['total_adopted_rows']}")
    print(f"Successfully generated: {stats['total_samples']}")
    print(f"Failed: {stats['failed_count']}")
    print(f"\nGenerated files:")
    print(f"  LQ images: {stats['total_lq']}")
    print(f"  REF images: {stats['total_ref']}")
    print(f"  LQ crop: {stats['total_lq_crop']}")
    print(f"  REF crop: {stats['total_ref_crop']}")
    print(f"  REF aligned: {stats['total_ref_aligned']}")
    print(f"  Prompts (legacy): {stats['total_prompt']}")
    print(f"  Prompts (optimized): {stats['total_prompt_opt']}")
    print(f"\nDataset index files:")
    print(f"  LQ: {stats['dataset_files']['lq_txt']}")
    print(f"  REF: {stats['dataset_files']['ref_txt']}")
    print(f"  LQ crop: {stats['dataset_files']['lq_crop_txt']}")
    print(f"  REF crop: {stats['dataset_files']['ref_crop_txt']}")
    print(f"  REF aligned: {stats['dataset_files']['ref_aligned_txt']}")
    print(f"  Prompt (legacy): {stats['dataset_files']['prompt_txt']}")
    print(f"  Prompt (optimized): {stats['dataset_files']['prompt_opt_txt']}")
    print(f"\nConfiguration:")
    print(f"  Grounding model: {stats['grounding_model']}")
    print(f"  Caption model: {stats['caption_model']}")
    print(f"  Border mode: {stats['border_mode']}")
    print(f"  Max aspect ratio change: {stats['max_aspect_ratio_change']}")
    print(f"\nOutput directory: {stats['output_dir']}")
    print(f"Statistics saved to: {os.path.join(args.output_dir, 'generation_stats.json')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
