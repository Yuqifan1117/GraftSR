#!/usr/bin/env python3
"""
生成测试数据集脚本
从标注数据生成测试集，包含：
- LQ图像（由HQ图像4倍下采样得到）
  - bicubic_lq: 简单bicubic下采样
  - degraded_lq: 复杂退化处理
- HQ图像（原图和裁剪图）
- REF图像（原图和裁剪图）
- Prompt文件
"""
import argparse
import json
import os
from typing import Dict, Any, List
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from convert_utils import process_image, create_prompt_file, write_image_paths_to_file
from client import Client

# 导入退化处理相关模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../diffsynth/extensions/realesrgan'))
from realesrgan import RealESRGAN_degradation
project_root = os.path.dirname(os.path.dirname(__file__))
degradation_config_path = os.path.join(project_root, 'examples/qwen_image/deg_pisa.yaml')


def downsample_image_bicubic(hq_image_path: str, scale_factor: int = 4) -> Image.Image:
    """
    对HQ图像进行bicubic下采样，生成LQ图像

    Args:
        hq_image_path: HQ图像路径
        scale_factor: 下采样倍数（默认4倍）

    Returns:
        下采样后的LQ图像对象
    """
    with Image.open(hq_image_path) as img:
        # 计算下采样后的尺寸
        width, height = img.size
        lq_width = width // scale_factor
        lq_height = height // scale_factor
        
        # 使用双三次插值下采样
        lq_img = img.resize((lq_width, lq_height), Image.BICUBIC)
        return lq_img

def downsample_image_degraded(hq_image_path: str, degradation_model, scale_factor: int = 4) -> Image.Image:
    """
    对HQ图像进行复杂退化处理，生成LQ图像

    Args:
        hq_image_path: HQ图像路径
        degradation_model: 退化处理模型
        scale_factor: 下采样倍数（默认4倍）

    Returns:
        退化后的LQ图像对象
    """
    # 读取图像并转换为numpy数组
    img = np.array(Image.open(hq_image_path).convert('RGB')).astype(np.float32) / 255.0
    
    # 使用退化模型处理
    with torch.no_grad():
        img_gt, img_lq = degradation_model.degrade_process(img)
    
    # 将tensor转换为PIL图像
    img_lq_np = img_lq.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_lq_np = (img_lq_np * 255.0).clip(0, 255).astype(np.uint8)
    lq_img = Image.fromarray(img_lq_np)
    
    return lq_img


def save_lq_image(lq_img: Image.Image, output_path: str) -> None:
    """
    保存LQ图像

    Args:
        lq_img: LQ图像对象
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lq_img.save(output_path, quality=95)


def process_test_data(annotation_data: Dict[str, Any],
                      output_img_dir: str,
                      output_prompt_dir: str,
                      min_image_side: int = 512,
                      scale_factor: int = 4) -> Dict[str, Any]:
    """
    处理单个标注数据，生成测试数据集

    Args:
        annotation_data: 标注数据
        output_img_dir: 输出图像目录
        output_prompt_dir: 输出提示词目录
        min_image_side: 图像最小短边要求
        scale_factor: 下采样倍数
        vlm_client: VLM Client 实例（启用 tile 纹理匹配时传入）
        tile_texture_model: tile 纹理匹配 VLM 模型名称

    Returns:
        处理结果信息
    """
    annotation = annotation_data.get("annotation", {})
    
    if not annotation:
        return {"error": "No annotation found", "item_id": annotation_data.get("item_id")}

    # 获取商品信息
    product_entity = annotation.get("product_entity", "")
    product_description = annotation.get("product_description", "")
    qualified_grounding_results = annotation.get("qualified_grounding_results", [])

    result = {
        "item_id": annotation_data.get("item_id"),
        "test_samples": [],  # 每个元素包含: hq, lq, ref, prompt等路径
        "filtered_samples": []
    }

    # 确保输出目录存在
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_prompt_dir, exist_ok=True)

    # 获取合格的图像（HQ图像）和参考图像
    qualified_image_paths = annotation.get("qualified_image_paths", [])
    
    # 处理参考图像（保存到ref子目录）
    ref_ori_dir = os.path.join(output_img_dir, "ref_ori")
    ref_crop_dir = os.path.join(output_img_dir, "ref_crop")
    
    output_original_ref_path = process_image(
        annotation.get("reference_image_path"), "ref_original", result["item_id"], ref_ori_dir,
        copy_images=True, min_image_side=min_image_side, filtered_pairs=result["filtered_samples"]
    ) if annotation.get("reference_image_path") else None
    
    output_cropped_ref_path = process_image(
        annotation.get("cropped_reference_image_path"), "ref_cropped", result["item_id"], ref_crop_dir,
        copy_images=True, min_image_side=min_image_side, filtered_pairs=result["filtered_samples"]
    ) if annotation.get("cropped_reference_image_path") else None
    
    # 参考原图和参考裁剪图都必须存在，否则不是有效样本
    if not output_original_ref_path or not output_cropped_ref_path:
        # 记录过滤原因
        if not output_original_ref_path:
            result["filtered_samples"].append({
                "reason": "original_ref_not_exist",
                "item_id": result["item_id"]
            })
        if not output_cropped_ref_path:
            result["filtered_samples"].append({
                "reason": "cropped_ref_not_exist",
                "item_id": result["item_id"]
            })
        return result

    # 初始化退化处理模型
    degradation_model = RealESRGAN_degradation(opt_path=degradation_config_path)

    # 为每张HQ图像生成对应的LQ图像和配对ref图像
    for idx, hq_img_path in enumerate(qualified_image_paths, start=1):
        # 通过索引获取对应的grounding_result（因为两者是按相同顺序生成的）
        current_grounding_result = None
        if idx - 1 < len(qualified_grounding_results):
            current_grounding_result = qualified_grounding_results[idx - 1]
        
        # 处理HQ原图（保存到hq_ori子目录）
        hq_ori_dir = os.path.join(output_img_dir, "hq_ori")
        output_hq_path = process_image(
            hq_img_path, "hq", result["item_id"], hq_ori_dir,
            copy_images=True, min_image_side=min_image_side, filtered_pairs=result["filtered_samples"], counter=idx
        )
        if not output_hq_path:
            continue
        
        # 生成LQ原图（由HQ原图下采样得到，保存到lq_ori_bicubic子目录）
        lq_img_bicubic = downsample_image_bicubic(hq_img_path, scale_factor)
        lq_ori_bicubic_dir = os.path.join(output_img_dir, "lq_ori_bicubic")
        output_lq_bicubic_path = os.path.join(lq_ori_bicubic_dir, os.path.basename(output_hq_path))
        save_lq_image(lq_img_bicubic, output_lq_bicubic_path)
        
        # 生成LQ原图（由HQ原图退化处理得到，保存到lq_ori_degraded子目录）
        lq_img_degraded = downsample_image_degraded(hq_img_path, degradation_model, scale_factor)
        lq_ori_degraded_dir = os.path.join(output_img_dir, "lq_ori_degraded")
        output_lq_degraded_path = os.path.join(lq_ori_degraded_dir, os.path.basename(output_hq_path))
        save_lq_image(lq_img_degraded, output_lq_degraded_path)
        
        # 处理HQ裁剪图（如果存在）
        output_cropped_hq_path = None
        output_cropped_lq_bicubic_path = None
        output_cropped_lq_degraded_path = None
        hq_spatial_description = None
        
        if current_grounding_result:
            # 从grounding_result中获取裁剪图像路径
            cropped_hq_img_path = current_grounding_result.get("cropped_image_path")
            hq_spatial_description = current_grounding_result.get("description")
            
            if cropped_hq_img_path and os.path.exists(cropped_hq_img_path):
                print(f"  [{idx}] Processing cropped image: {os.path.basename(cropped_hq_img_path)}")
                # 处理HQ裁剪图（保存到hq_crop子目录）
                hq_crop_dir = os.path.join(output_img_dir, "hq_crop")
                output_cropped_hq_path = process_image(
                    cropped_hq_img_path, "cropped_hq", result["item_id"], hq_crop_dir,
                    copy_images=True, min_image_side=min_image_side, filtered_pairs=result["filtered_samples"], counter=idx, gt_path=hq_img_path
                )
                
                if output_cropped_hq_path:
                    print(f"  [{idx}] Successfully processed cropped HQ image")
                    # 生成LQ裁剪图（由HQ裁剪图下采样得到，保存到lq_crop_bicubic子目录）
                    cropped_lq_img_bicubic = downsample_image_bicubic(cropped_hq_img_path, scale_factor)
                    lq_crop_bicubic_dir = os.path.join(output_img_dir, "lq_crop_bicubic")
                    output_cropped_lq_bicubic_path = os.path.join(lq_crop_bicubic_dir, os.path.basename(output_cropped_hq_path))
                    save_lq_image(cropped_lq_img_bicubic, output_cropped_lq_bicubic_path)
                    
                    # 生成LQ裁剪图（由HQ裁剪图退化处理得到，保存到lq_crop_degraded子目录）
                    cropped_lq_img_degraded = downsample_image_degraded(cropped_hq_img_path, degradation_model, scale_factor)
                    lq_crop_degraded_dir = os.path.join(output_img_dir, "lq_crop_degraded")
                    output_cropped_lq_degraded_path = os.path.join(lq_crop_degraded_dir, os.path.basename(output_cropped_hq_path))
                    save_lq_image(cropped_lq_img_degraded, output_cropped_lq_degraded_path)
                else:
                    print(f"  [{idx}] WARNING: Failed to process cropped image (size too small or processing error)")
            else:
                if not cropped_hq_img_path:
                    print(f"  [{idx}] WARNING: No cropped_image_path in grounding_result")
                else:
                    print(f"  [{idx}] WARNING: Cropped image file does not exist: {cropped_hq_img_path}")
        else:
            print(f"  [{idx}] WARNING: No grounding_result found for this HQ image")
        
        # 如果没有有效的裁剪图，跳过该样本（因为需要裁剪图配对）
        if not output_cropped_hq_path:
            print(f"  [{idx}] Skipping sample: no valid cropped image")
            # 删除已生成的HQ和LQ原图
            if os.path.exists(output_hq_path):
                os.remove(output_hq_path)
            if os.path.exists(output_lq_bicubic_path):
                os.remove(output_lq_bicubic_path)
            if os.path.exists(output_lq_degraded_path):
                os.remove(output_lq_degraded_path)
            continue
        
        # 为HQ原图创建prompt文件
        prompt_file = create_prompt_file(output_hq_path, product_entity, product_description, hq_spatial_description)
        
        # 为HQ裁剪图创建prompt文件
        cropped_prompt_file = create_prompt_file(output_cropped_hq_path, product_entity, product_description)
        
        # 检查prompt是否创建成功
        if not prompt_file or not cropped_prompt_file:
            print(f"Filtering sample: no valid prompt for {output_hq_path} or {output_cropped_hq_path}")
            result["filtered_samples"].append({
                "reason": "no_valid_prompt",
                "hq_path": output_hq_path,
                "cropped_hq": output_cropped_hq_path
            })
            # 删除已生成的文件
            for path in [output_hq_path, output_lq_bicubic_path, output_lq_degraded_path, 
                         output_cropped_hq_path, output_cropped_lq_bicubic_path, output_cropped_lq_degraded_path]:
                if path and os.path.exists(path):
                    os.remove(path)
            if prompt_file and os.path.exists(prompt_file):
                os.remove(prompt_file)
            if cropped_prompt_file and os.path.exists(cropped_prompt_file):
                os.remove(cropped_prompt_file)
            continue

        # 添加测试样本
        sample = {
            "hq_ori": output_hq_path,
            "lq_ori_bicubic": output_lq_bicubic_path,
            "lq_ori_degraded": output_lq_degraded_path,
            "hq_crop": output_cropped_hq_path,
            "lq_crop_bicubic": output_cropped_lq_bicubic_path,
            "lq_crop_degraded": output_cropped_lq_degraded_path,
            "ref_ori": output_original_ref_path,
            "ref_crop": output_cropped_ref_path,
            "prompt_ori": prompt_file,
            "prompt_crop": cropped_prompt_file
        }
        result["test_samples"].append(sample)

    return result


def generate_test_dataset(annotation_list_json: str,
                         output_dir: str,
                         min_image_side: int = 512,
                         scale_factor: int = 4,
                         max_samples: int = None,
                         enable_tile_texture: bool = False,
                         tile_texture_model: str = "qwen3.5-plus") -> Dict[str, Any]:
    """
    生成测试数据集

    Args:
        annotation_list_json: 标注列表JSON文件路径
        output_dir: 输出目录
        min_image_side: 图像最小短边要求
        scale_factor: 下采样倍数
        max_samples: 最多处理N条标注（默认None，表示不限制）

    Returns:
        生成统计信息
    """
    # 创建输出目录结构
    output_img_dir = os.path.join(output_dir, "img")
    output_prompt_dir = os.path.join(output_dir, "prompt")
    
    # 创建各子目录
    lq_ori_dir = os.path.join(output_img_dir, "lq_ori_bicubic")
    lq_crop_dir = os.path.join(output_img_dir, "lq_crop_bicubic")
    lq_ori_degraded_dir = os.path.join(output_img_dir, "lq_ori_degraded")
    lq_crop_degraded_dir = os.path.join(output_img_dir, "lq_crop_degraded")
    hq_ori_dir = os.path.join(output_img_dir, "hq_ori")
    hq_crop_dir = os.path.join(output_img_dir, "hq_crop")
    ref_ori_dir = os.path.join(output_img_dir, "ref_ori")
    ref_crop_dir = os.path.join(output_img_dir, "ref_crop")
    prompt_ori_dir = os.path.join(output_prompt_dir, "prompt_ori")
    prompt_crop_dir = os.path.join(output_prompt_dir, "prompt_crop")
    
    for dir_path in [lq_ori_dir, lq_crop_dir, lq_ori_degraded_dir, lq_crop_degraded_dir,
                     hq_ori_dir, hq_crop_dir, 
                     ref_ori_dir, ref_crop_dir,
                     prompt_ori_dir, prompt_crop_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # 读取标注列表JSON文件
    with open(annotation_list_json, 'r', encoding='utf-8') as f:
        annotation_list = json.load(f)

    print(f"Found {len(annotation_list)} annotation entries")
    
    # 应用限制（测试集只处理开头的N条）
    if max_samples is not None and max_samples > 0:
        print(f"Limiting to first {max_samples} annotations")
        annotation_list = annotation_list[:max_samples]
    
    print(f"Processing {len(annotation_list)} annotations")

    # 处理每个标注条目
    all_results = []
    
    # 收集各类图像路径
    lq_ori_bicubic_paths = []
    lq_crop_bicubic_paths = []
    lq_ori_degraded_paths = []
    lq_crop_degraded_paths = []
    hq_ori_paths = []
    hq_crop_paths = []
    ref_ori_paths = []
    ref_crop_paths = []
    prompt_ori_paths = []
    prompt_crop_paths = []

    # 过滤统计
    filter_stats = {
        "ref_too_small": 0,
        "ref_not_exist": 0,
        "hq_too_small": 0,
        "hq_not_exist": 0,
        "cropped_hq_too_small": 0,
        "cropped_hq_not_exist": 0,
        "no_valid_prompt": 0,
        "total_filtered": 0
    }

    for annotation_data in tqdm(annotation_list, desc="Generating test dataset"):
        try:
            result = process_test_data(
                annotation_data,
                output_img_dir,
                output_prompt_dir,
                min_image_side,
                scale_factor,
            )
            all_results.append(result)

            # 收集过滤统计
            for filtered_sample in result.get("filtered_samples", []):
                filter_stats["total_filtered"] += 1
                reason = filtered_sample.get("reason", "")
                if reason == "ref_too_small":
                    filter_stats["ref_too_small"] += 1
                elif reason == "ref_not_exist":
                    filter_stats["ref_not_exist"] += 1
                elif reason == "hq_too_small":
                    filter_stats["hq_too_small"] += 1
                elif reason == "hq_not_exist":
                    filter_stats["hq_not_exist"] += 1
                elif reason == "cropped_hq_too_small":
                    filter_stats["cropped_hq_too_small"] += 1
                elif reason == "cropped_hq_not_exist":
                    filter_stats["cropped_hq_not_exist"] += 1
                elif reason == "no_valid_prompt":
                    filter_stats["no_valid_prompt"] += 1

            # 收集图像路径（现在所有样本都保证有完整字段）
            for sample in result.get("test_samples", []):
                lq_ori_bicubic_paths.append(sample["lq_ori_bicubic"])
                lq_crop_bicubic_paths.append(sample["lq_crop_bicubic"])
                lq_ori_degraded_paths.append(sample["lq_ori_degraded"])
                lq_crop_degraded_paths.append(sample["lq_crop_degraded"])
                hq_ori_paths.append(sample["hq_ori"])
                hq_crop_paths.append(sample["hq_crop"])
                # 移除 if 判断，直接添加（因为现在保证都存在）
                ref_ori_paths.append(sample["ref_ori"])
                ref_crop_paths.append(sample["ref_crop"])
                prompt_ori_paths.append(sample["prompt_ori"])
                prompt_crop_paths.append(sample["prompt_crop"])

        except Exception as e:
            print(f"Error processing {annotation_data.get('item_id', 'unknown')}: {e}")
            continue

    # 写入数据集文本文件
    lq_ori_txt = os.path.join(output_dir, "lq_ori_bicubic.txt")
    lq_crop_txt = os.path.join(output_dir, "lq_crop_bicubic.txt")
    lq_ori_degraded_txt = os.path.join(output_dir, "lq_ori_degraded.txt")
    lq_crop_degraded_txt = os.path.join(output_dir, "lq_crop_degraded.txt")
    hq_ori_txt = os.path.join(output_dir, "hq_ori.txt")
    hq_crop_txt = os.path.join(output_dir, "hq_crop.txt")
    ref_ori_txt = os.path.join(output_dir, "ref_ori.txt")
    ref_crop_txt = os.path.join(output_dir, "ref_crop.txt")
    prompt_ori_txt = os.path.join(output_dir, "prompt_ori.txt")
    prompt_crop_txt = os.path.join(output_dir, "prompt_crop.txt")
    
    write_image_paths_to_file(lq_ori_txt, lq_ori_bicubic_paths)
    write_image_paths_to_file(lq_crop_txt, lq_crop_bicubic_paths)
    write_image_paths_to_file(lq_ori_degraded_txt, lq_ori_degraded_paths)
    write_image_paths_to_file(lq_crop_degraded_txt, lq_crop_degraded_paths)
    write_image_paths_to_file(hq_ori_txt, hq_ori_paths)
    write_image_paths_to_file(hq_crop_txt, hq_crop_paths)
    write_image_paths_to_file(ref_ori_txt, ref_ori_paths)
    write_image_paths_to_file(ref_crop_txt, ref_crop_paths)
    write_image_paths_to_file(prompt_ori_txt, prompt_ori_paths)
    write_image_paths_to_file(prompt_crop_txt, prompt_crop_paths)

    # 统计信息
    stats = {
        "total_annotations": len(annotation_list),
        "processed_annotations": len(all_results),
        "total_samples": len(hq_ori_paths),
        "total_lq_ori_bicubic": len(lq_ori_bicubic_paths),
        "total_lq_crop_bicubic": len(lq_crop_bicubic_paths),
        "total_lq_ori_degraded": len(lq_ori_degraded_paths),
        "total_lq_crop_degraded": len(lq_crop_degraded_paths),
        "total_hq_ori": len(hq_ori_paths),
        "total_hq_crop": len(hq_crop_paths),
        "total_ref_ori": len(ref_ori_paths),
        "total_ref_crop": len(ref_crop_paths),
        "total_prompt_ori": len(prompt_ori_paths),
        "total_prompt_crop": len(prompt_crop_paths),
        "output_dir": output_dir,
        "scale_factor": scale_factor,
        "min_image_side": min_image_side,
        "filter_stats": filter_stats,
        "dataset_files": {
            "lq_ori_txt": lq_ori_txt,
            "lq_crop_txt": lq_crop_txt,
            "lq_ori_degraded_txt": lq_ori_degraded_txt,
            "lq_crop_degraded_txt": lq_crop_degraded_txt,
            "hq_ori_txt": hq_ori_txt,
            "hq_crop_txt": hq_crop_txt,
            "ref_ori_txt": ref_ori_txt,
            "ref_crop_txt": ref_crop_txt,
            "prompt_ori_txt": prompt_ori_txt,
            "prompt_crop_txt": prompt_crop_txt
        }
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate test dataset from annotations")
    parser.add_argument("--annotation-list-json", type=str, required=True,
                        help="Path to annotation list JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for test data")
    parser.add_argument("--min-image-side", type=int, default=512,
                        help="Minimum side length for images (default: 512)")
    parser.add_argument("--scale-factor", type=int, default=4,
                        help="Downsampling scale factor for LQ images (default: 4)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to the first N samples (default: None)")
    args = parser.parse_args()

    # 执行生成
    stats = generate_test_dataset(
        annotation_list_json=args.annotation_list_json,
        output_dir=args.output_dir,
        min_image_side=args.min_image_side,
        scale_factor=args.scale_factor,
        max_samples=args.max_samples,
    )

    # 打印统计信息
    print("\n" + "=" * 60)
    print("Test dataset generation completed!")
    print("=" * 60)
    print(f"Total annotation entries: {stats['total_annotations']}")
    print(f"Successfully processed: {stats['processed_annotations']}")
    print(f"Total test samples: {stats['total_samples']}")
    print(f"\nGenerated images:")
    print(f"  LQ original bicubic: {stats['total_lq_ori_bicubic']}")
    print(f"  LQ cropped bicubic: {stats['total_lq_crop_bicubic']}")
    print(f"  LQ original degraded: {stats['total_lq_ori_degraded']}")
    print(f"  LQ cropped degraded: {stats['total_lq_crop_degraded']}")
    print(f"  HQ original: {stats['total_hq_ori']}")
    print(f"  HQ cropped: {stats['total_hq_crop']}")
    print(f"  REF original: {stats['total_ref_ori']}")
    print(f"  REF cropped: {stats['total_ref_crop']}")
    print(f"\nGenerated prompts:")
    print(f"  Prompt original: {stats['total_prompt_ori']}")
    print(f"  Prompt cropped: {stats['total_prompt_crop']}")
    print(f"\nConfiguration:")
    print(f"  Downsampling scale factor: {stats['scale_factor']}x")
    print(f"  Image filter threshold: min side >= {stats['min_image_side']}")
    print(f"\nFiltered samples statistics:")
    print(f"  Reference too small: {stats['filter_stats']['ref_too_small']}")
    print(f"  Reference not exist: {stats['filter_stats']['ref_not_exist']}")
    print(f"  HQ too small: {stats['filter_stats']['hq_too_small']}")
    print(f"  HQ not exist: {stats['filter_stats']['hq_not_exist']}")
    print(f"  Cropped HQ too small: {stats['filter_stats']['cropped_hq_too_small']}")
    print(f"  Cropped HQ not exist: {stats['filter_stats']['cropped_hq_not_exist']}")
    print(f"  No valid prompt: {stats['filter_stats']['no_valid_prompt']}")
    print(f"  Total filtered: {stats['filter_stats']['total_filtered']}")
    print(f"\nOutput directory: {stats['output_dir']}")
    print(f"\nDataset text files:")
    print(f"  LQ original bicubic: {stats['dataset_files']['lq_ori_txt']}")
    print(f"  LQ cropped bicubic: {stats['dataset_files']['lq_crop_txt']}")
    print(f"  LQ original degraded: {stats['dataset_files']['lq_ori_degraded_txt']}")
    print(f"  LQ cropped degraded: {stats['dataset_files']['lq_crop_degraded_txt']}")
    print(f"  HQ original: {stats['dataset_files']['hq_ori_txt']}")
    print(f"  HQ cropped: {stats['dataset_files']['hq_crop_txt']}")
    print(f"  REF original: {stats['dataset_files']['ref_ori_txt']}")
    print(f"  REF cropped: {stats['dataset_files']['ref_crop_txt']}")
    print(f"  Prompt original: {stats['dataset_files']['prompt_ori_txt']}")
    print(f"  Prompt cropped: {stats['dataset_files']['prompt_crop_txt']}")
    print("=" * 60)

    # 保存统计信息
    stats_json_path = os.path.join(args.output_dir, "generation_stats.json")
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved to: {stats_json_path}")


if __name__ == "__main__":
    main()