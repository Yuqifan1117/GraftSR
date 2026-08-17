import os
import shutil
from typing import Tuple, List, Dict, Optional, Any
from PIL import Image


def check_image_min_side(image_path: str, min_side: int = 512) -> Tuple[bool, int, int]:
    """
    检查图像短边是否满足最小尺寸要求

    Args:
        image_path: 图像路径
        min_side: 最小短边尺寸要求

    Returns:
        (是否满足要求, 宽度, 高度)
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            min_dimension = min(width, height)
            return min_dimension >= min_side, width, height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return False, 0, 0


def create_prompt_file(image_path: str,
                       product_entity: str,
                       product_description: str,
                       spatial_grounding_description: str = None) -> str:
    """
    为图像创建对应的文本提示词文件

    Args:
        image_path: 图像路径
        product_entity: 商品主体词
        product_description: 商品描述
        spatial_grounding_description: 空间定位描述（仅用于GT原图）

    Returns:
        提示词文件路径，如果无法创建有效prompt则返回None
    """
    # 检查是否有有效的内容
    if not product_entity or not product_description:
        print(f"Warning: No valid prompt content for image {image_path}")
        return None

    # 根据图像路径生成提示词文件路径
    # 将 /img/ 替换为 /prompt/，扩展名替换为 .txt
    prompt_path = image_path.replace("/img/", "/prompt/")
    prompt_path = os.path.splitext(prompt_path)[0] + ".txt"

    # 确保目录存在
    os.makedirs(os.path.dirname(prompt_path), exist_ok=True)

    # 构建提示词内容
    if spatial_grounding_description:
        # GT原图的prompt格式：f'{spatial_grounding_description} {product_entity}：{product_description}'
        prompt_text = f"{spatial_grounding_description} {product_entity}：{product_description}"
    else:
        # GT裁剪图的prompt格式：f'{product_entity}：{product_description}'
        prompt_text = f"{product_entity}：{product_description}"

    # 写入提示词文件
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt_text)

    return prompt_path


def write_image_paths_to_file(file_path: str, image_paths: List[str]) -> None:
    """将图像路径列表写入文本文件"""
    if not image_paths:
        return
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(image_paths) + '\n')


def process_image(img_path: str,
                  img_type: str,
                  item_id: str,
                  output_img_dir: str,
                  copy_images: bool,
                  min_image_side: int,
                  filtered_pairs: List[Dict[str, Any]],
                  counter: int = None,
                  gt_path: str = None) -> Optional[str]:
    """处理图像：检查存在性、尺寸，并复制图像

    Args:
        img_path: 图像路径
        img_type: 图像类型（gt, cropped_gt, ref_original, ref_cropped）
        item_id: 商品ID
        output_img_dir: 输出目录
        copy_images: 是否复制图像
        min_image_side: 最小边长要求
        filtered_pairs: 过滤记录列表
        counter: 计数器（用于GT图像生成唯一文件名）
        gt_path: GT图像路径（仅用于裁剪GT时记录）

    Returns:
        处理后的图像路径，失败返回None
    """
    # 根据 img_type 映射到原有的 reason 名称
    reason_map = {
        "gt": ("gt_not_exist", "gt_image_too_small", "gt_path"),
        "cropped_gt": ("cropped_gt_not_exist", "cropped_gt_too_small", "cropped_gt_path"),
        "ref_original": ("original_ref_not_exist", "original_ref_too_small", "ref_path"),
        "ref_cropped": ("cropped_ref_not_exist", "cropped_ref_too_small", "ref_path")
    }
    
    not_exist_reason, too_small_reason, path_field = reason_map.get(img_type, (f"{img_type}_not_exist", f"{img_type}_too_small", "img_path"))

    if not os.path.exists(img_path):
        print(f"Warning: {img_type.replace('_', ' ').capitalize()} image {img_path} does not exist")
        filtered_pairs.append({
            "reason": not_exist_reason,
            path_field: img_path
        })
        # 对于裁剪GT，额外记录 gt_path
        if img_type == "cropped_gt" and gt_path:
            filtered_pairs[-1]["gt_path"] = gt_path
        return None

    # 检查图像尺寸
    img_valid, img_width, img_height = check_image_min_side(img_path, min_image_side)
    if not img_valid:
        print(f"Filtering {img_type} image {img_path}: min side {min(img_width, img_height)} < {min_image_side}")
        filtered_pairs.append({
            "reason": too_small_reason,
            path_field: img_path,
            "img_size": (img_width, img_height)
        })
        # 对于裁剪GT，额外记录 gt_path
        if img_type == "cropped_gt" and gt_path:
            filtered_pairs[-1]["gt_path"] = gt_path
        return None

    # 复制或返回原始路径
    if copy_images:
        img_filename = os.path.basename(img_path)
        if counter is not None:
            if img_type == "cropped_gt":
                # 裁剪GT的特殊命名格式：{item_id}_gt{counter}_cropped_{img_filename}
                new_img_filename = f"{item_id}_gt{counter}_cropped_{img_filename}"
            else:
                new_img_filename = f"{item_id}_{img_type}{counter}_{img_filename}"
        else:
            new_img_filename = f"{item_id}_{img_type}_{img_filename}"
        output_img_path = os.path.join(output_img_dir, new_img_filename)
        shutil.copy2(img_path, output_img_path)
        return output_img_path
    return img_path
