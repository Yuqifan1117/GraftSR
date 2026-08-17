# !/usr/bin/env python3
"""
将标注结果转换为训练集格式
支持生成：
1. GT图像路径文本文件（hq_ori.txt）
2. 裁剪参考图像路径文本文件（ref_crop.txt）
3. 对应的文本提示词文件路径（prompt_ori.txt）

HQ-Ref 配对需满足 embedding 相似度不低于阈值（默认 0.85），低于阈值的配对直接过滤，
相似度本身不作为输出文件。
"""
import argparse
import json
import os
from typing import Dict, Any
from tqdm import tqdm

from convert_utils import process_image, create_prompt_file, write_image_paths_to_file


def process_single_annotation(annotation_data: Dict[str, Any],
                              output_img_dir: str,
                              output_prompt_dir: str,
                              copy_images: bool = True,
                              min_image_side: int = 512,
                              similarity_threshold: float = 0.85) -> Dict[str, Any]:
    """
    处理单个标注结果，为每张GT图像配对ref图像和prompt
    只保留GT-REF-TEXT三元组完整、图像尺寸满足要求、且HQ-Ref相似度不低于阈值的样本

    Args:
        annotation_data: 标注数据（从annotation_list_json中读取的单个条目）
        output_img_dir: 输出图像目录
        output_prompt_dir: 输出提示词目录
        copy_images: 是否复制图像到输出目录
        min_image_side: 图像最小短边要求（用于过滤）
        similarity_threshold: HQ-Ref embedding 相似度下限（用于过滤）

    Returns:
        处理结果信息，包含gt_ref_pairs列表和过滤统计
    """
    # 提取标注信息
    annotation = annotation_data.get("annotation", {})

    if not annotation:
        return {"error": "No annotation found", "item_id": annotation_data.get("item_id")}

    # 获取商品信息
    product_entity = annotation.get("product_entity", "")
    product_description = annotation.get("product_description", "")

    # 获取合格图像的grounding结果，用于提取spatial_grounding_description
    qualified_grounding_results = annotation.get("qualified_grounding_results", [])

    # 获取合格图像与参考图像之间的embedding相似度
    qualified_similarities = annotation.get("qualified_similarities", {})

    result = {
        "item_id": annotation_data.get("item_id"),
        "gt_ref_pairs": [],  # 每个元素包含: {"gt": gt_path, "cropped_ref": path, "prompt": prompt_path}
        "filtered_pairs": []  # 被过滤掉的配对信息
    }

    # 确保输出目录存在
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_prompt_dir, exist_ok=True)

    # 获取合格的图像（GT图像）
    qualified_image_paths = annotation.get("qualified_image_paths", [])

    # 处理裁剪参考图（ref_crop）
    output_cropped_ref_path = process_image(
        annotation.get("cropped_reference_image_path"), "ref_cropped", result["item_id"], output_img_dir,
        copy_images, min_image_side, result["filtered_pairs"]
    ) if annotation.get("cropped_reference_image_path") else None

    # 裁剪参考图必须存在，否则不是有效样本
    if not output_cropped_ref_path:
        result["filtered_pairs"].append({
            "reason": "cropped_ref_not_exist",
            "item_id": result["item_id"]
        })
        return result

    # 为每张GT图像配对ref图像和prompt
    for idx, gt_img_path in enumerate(qualified_image_paths, start=1):
        # 基于embedding相似度过滤：缺失或低于阈值的配对直接跳过
        pair_similarity = qualified_similarities.get(gt_img_path, None)
        if pair_similarity is None:
            result["filtered_pairs"].append({
                "reason": "missing_similarity",
                "gt_path": gt_img_path
            })
            continue
        if pair_similarity < similarity_threshold:
            result["filtered_pairs"].append({
                "reason": "low_similarity",
                "gt_path": gt_img_path,
                "similarity": pair_similarity
            })
            continue

        # 处理GT图像
        output_gt_path = process_image(
            gt_img_path, "gt", result["item_id"], output_img_dir,
            copy_images, min_image_side, result["filtered_pairs"], len(result["gt_ref_pairs"]) + 1,
        )
        if not output_gt_path:
            continue

        # 从grounding_result中获取空间定位描述，用于构建prompt（按索引对应，顺序一致）
        gt_spatial_description = None
        if idx - 1 < len(qualified_grounding_results):
            gt_spatial_description = qualified_grounding_results[idx - 1].get("description")

        # 为GT图像创建对应的prompt文件
        prompt_file = create_prompt_file(output_gt_path, product_entity, product_description, gt_spatial_description)

        # 检查prompt是否创建成功
        if not prompt_file:
            print(f"Filtering GT-Ref pair: no valid prompt for {output_gt_path}")
            result["filtered_pairs"].append({
                "reason": "no_valid_prompt",
                "gt_path": output_gt_path,
                "cropped_ref": output_cropped_ref_path
            })
            # 删除已复制的GT图像和已创建的prompt文件
            if copy_images:
                if os.path.exists(output_gt_path):
                    os.remove(output_gt_path)
                if prompt_file and os.path.exists(prompt_file):
                    os.remove(prompt_file)
            continue

        # 添加配对结果 - GT-CROPPED_REF-TEXT三元组
        result["gt_ref_pairs"].append({
            "gt": output_gt_path,
            "cropped_ref": output_cropped_ref_path,
            "prompt": prompt_file,
        })

    return result


def convert_annotations_to_training(annotation_list_json: str,
                                    output_dir: str,
                                    copy_images: bool = True,
                                    min_image_side: int = 512,
                                    similarity_threshold: float = 0.85,
                                    skip_first: int = 0,
                                    max_samples: int = None) -> Dict[str, Any]:
    """
    将标注结果转换为训练集
    只保留GT-REF-TEXT三元组完整、图像尺寸满足要求、且HQ-Ref相似度不低于阈值的样本

    Args:
        annotation_list_json: 标注列表JSON文件路径
        output_dir: 输出目录
        copy_images: 是否复制图像到输出目录
        min_image_side: 图像最小短边要求（用于过滤）
        similarity_threshold: HQ-Ref embedding 相似度下限（用于过滤）
        skip_first: 跳过开头的N条标注（默认0）
        max_samples: 最多处理N条标注（默认None，表示不限制）

    Returns:
        转换统计信息
    """
    # 创建输出目录结构
    output_img_dir = os.path.join(output_dir, "img")
    output_prompt_dir = os.path.join(output_dir, "prompt")

    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_prompt_dir, exist_ok=True)

    # 自动生成数据集文本文件路径
    main_dataset_txt = os.path.join(output_dir, "hq_ori.txt")
    cropped_ref_dataset_txt = os.path.join(output_dir, "ref_crop.txt")
    prompt_ori_txt = os.path.join(output_dir, "prompt_ori.txt")

    # 读取标注列表JSON文件
    with open(annotation_list_json, 'r', encoding='utf-8') as f:
        annotation_list = json.load(f)

    print(f"Found {len(annotation_list)} annotation entries")

    # 应用跳过和限制
    if skip_first > 0:
        print(f"Skipping first {skip_first} annotations")
        annotation_list = annotation_list[skip_first:]

    if max_samples is not None and max_samples > 0:
        print(f"Limiting to {max_samples} annotations")
        annotation_list = annotation_list[:max_samples]

    print(f"Processing {len(annotation_list)} annotations")

    # 处理每个标注条目
    all_results = []
    gt_image_paths = []
    cropped_ref_image_paths = []
    prompt_ori_paths = []

    # 过滤统计
    filter_stats = {
        "cropped_ref_too_small": 0,
        "cropped_ref_not_exist": 0,
        "gt_image_too_small": 0,
        "gt_not_exist": 0,
        "missing_similarity": 0,
        "low_similarity": 0,
        "no_valid_prompt": 0,
        "total_filtered_pairs": 0
    }

    for annotation_data in tqdm(annotation_list, desc="Processing annotations"):
        try:
            result = process_single_annotation(
                annotation_data,
                output_img_dir,
                output_prompt_dir,
                copy_images,
                min_image_side,
                similarity_threshold
            )
            all_results.append(result)

            # 收集过滤统计
            for filtered_pair in result.get("filtered_pairs", []):
                filter_stats["total_filtered_pairs"] += 1
                reason = filtered_pair.get("reason", "")
                if reason in filter_stats:
                    filter_stats[reason] += 1

            # 收集图像路径和prompt路径
            for pair in result.get("gt_ref_pairs", []):
                gt_image_paths.append(pair["gt"])
                cropped_ref_image_paths.append(pair["cropped_ref"])
                prompt_ori_paths.append(pair["prompt"])

        except Exception as e:
            print(f"Error processing {annotation_data.get('item_id', 'unknown')}: {e}")
            continue

    # 写入数据集文本文件
    write_image_paths_to_file(main_dataset_txt, gt_image_paths)
    write_image_paths_to_file(cropped_ref_dataset_txt, cropped_ref_image_paths)
    write_image_paths_to_file(prompt_ori_txt, prompt_ori_paths)

    # 统计信息
    stats = {
        "total_annotations": len(annotation_list),
        "processed_annotations": len(all_results),
        "total_gt_images": len(gt_image_paths),
        "total_cropped_ref_images": len(cropped_ref_image_paths),
        "total_prompt_files": len(prompt_ori_paths),
        "total_gt_ref_pairs": sum(len(r.get("gt_ref_pairs", [])) for r in all_results),
        "output_img_dir": output_img_dir,
        "output_prompt_dir": output_prompt_dir,
        "main_dataset_txt": main_dataset_txt,
        "cropped_ref_dataset_txt": cropped_ref_dataset_txt if cropped_ref_image_paths else None,
        "prompt_ori_txt": prompt_ori_txt if prompt_ori_paths else None,
        "filter_stats": filter_stats,
        "min_image_side": min_image_side,
        "similarity_threshold": similarity_threshold
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert annotation results to training dataset")
    parser.add_argument("--annotation-list-json", type=str, required=True,
                        help="Path to annotation list JSON file")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for training data")
    parser.add_argument("--min-image-side", type=int, default=512,
                        help="Minimum side length for images (default: 512)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85,
                        help="Minimum HQ-Ref embedding similarity for keeping a pair (default: 0.85)")
    parser.add_argument("--skip-first", type=int, default=0,
                        help="Skip the first N annotations (default: 0)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit to the first N samples (default: None)")
    args = parser.parse_args()

    # 执行转换
    stats = convert_annotations_to_training(
        annotation_list_json=args.annotation_list_json,
        output_dir=args.output_dir,
        copy_images=True,
        min_image_side=args.min_image_side,
        similarity_threshold=args.similarity_threshold,
        skip_first=args.skip_first,
        max_samples=args.max_samples
    )

    # 打印统计信息
    print("\n" + "=" * 50)
    print("Conversion completed!")
    print("=" * 50)
    print(f"Total annotation entries: {stats['total_annotations']}")
    print(f"Successfully processed: {stats['processed_annotations']}")
    print(f"Total GT images: {stats['total_gt_images']}")
    print(f"Total cropped reference images: {stats['total_cropped_ref_images']}")
    print(f"Total GT-Ref pairs: {stats['total_gt_ref_pairs']}")
    print(f"Total prompt files: {stats['total_prompt_files']}")
    print(f"Image filter threshold: min side >= {stats['min_image_side']}")
    print(f"Similarity filter threshold: >= {stats['similarity_threshold']}")
    print(f"\nFiltered pairs statistics:")
    print(f"  Cropped reference too small: {stats['filter_stats']['cropped_ref_too_small']}")
    print(f"  Cropped reference not exist: {stats['filter_stats']['cropped_ref_not_exist']}")
    print(f"  GT too small: {stats['filter_stats']['gt_image_too_small']}")
    print(f"  GT not exist: {stats['filter_stats']['gt_not_exist']}")
    print(f"  Missing similarity: {stats['filter_stats']['missing_similarity']}")
    print(f"  Low similarity: {stats['filter_stats']['low_similarity']}")
    print(f"  No valid prompt: {stats['filter_stats']['no_valid_prompt']}")
    print(f"  Total filtered pairs: {stats['filter_stats']['total_filtered_pairs']}")
    print(f"\nOutput directories:")
    print(f"  Images: {stats['output_img_dir']}")
    print(f"  Prompts: {stats['output_prompt_dir']}")
    print(f"\nDataset text files:")
    print(f"  Main dataset: {stats['main_dataset_txt']}")
    if stats['cropped_ref_dataset_txt']:
        print(f"  Cropped reference dataset: {stats['cropped_ref_dataset_txt']}")
    if stats['prompt_ori_txt']:
        print(f"  Prompt dataset: {stats['prompt_ori_txt']}")
    print("=" * 50)

    # 保存统计信息
    stats_json_path = os.path.join(args.output_dir, "conversion_stats.json")
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved to: {stats_json_path}")


if __name__ == "__main__":
    main()
