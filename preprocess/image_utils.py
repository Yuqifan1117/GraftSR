import math
import logging
import base64
import mimetypes
import io
import os
from typing import Dict, Any, List, Literal

import requests
from PIL import Image

logger = logging.getLogger(__name__)

QWEN25VL_IMAGE_FACTOR = 28
QWEN25VL_MIN_PIXELS = 4 * 28 * 28
QWEN25VL_MAX_PIXELS = 1280 * 28 * 28
MAX_RATIO = 200


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def qwenvl_smart_resize(
    height: int, width: int, factor: int = QWEN25VL_IMAGE_FACTOR, min_pixels: int = QWEN25VL_MIN_PIXELS, max_pixels: int = QWEN25VL_MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def resize_vlm_bboxes(bounding_boxes: List[Dict[str, Any]], height: int, width: int,
                      version: Literal['qwen25vl', 'qwen3vl', 'qwen3.5', 'qwen3.6', 'gemini'] = "qwen25vl") -> List[Dict[str, Any]]:
    """
    Resizes bounding boxes from the model's scaled input dimensions back to the original image dimensions.
    """
    if version == "qwen25vl":
        input_height, input_width = qwenvl_smart_resize(height, width, QWEN25VL_IMAGE_FACTOR,
                                                        QWEN25VL_MIN_PIXELS, QWEN25VL_MAX_PIXELS)
    elif version == "qwen3vl" or version == "qwen3.5" or version == "qwen3.6":
        input_height, input_width = 1000, 1000
    elif version == "gemini":
        input_height, input_width = 1000, 1000
        for i, bounding_box in enumerate(bounding_boxes):
            box_coords = bounding_box["bbox_2d"]
            bounding_box["bbox_2d"] = [box_coords[1], box_coords[0], box_coords[3], box_coords[2]]
    else:
        input_height, input_width = height, width

    logger.debug(f'Resize bbox from {input_width}x{input_height} to {width}x{height}')
    for i, bounding_box in enumerate(bounding_boxes):
        # Convert normalized coordinates to absolute coordinates
        box_coords = bounding_box["bbox_2d"]
        abs_x1 = int(box_coords[0] / input_width * width)
        abs_y1 = int(box_coords[1] / input_height * height)
        abs_x2 = int(box_coords[2] / input_width * width)
        abs_y2 = int(box_coords[3] / input_height * height)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        bounding_box["bbox_2d"] = [abs_x1, abs_y1, abs_x2, abs_y2]

    max_x = max(box["bbox_2d"][2] for box in bounding_boxes)
    if max_x > width:
        logger.warning(f'Max bbox x {max_x} exceeds image width {width}. Rescaling x-coordinates.')
        x_scale_factor = width / max_x
        for bounding_box in bounding_boxes:
            box_coords = bounding_box["bbox_2d"]
            scaled_x1 = int(box_coords[0] * x_scale_factor)
            scaled_x2 = int(box_coords[2] * x_scale_factor)
            bounding_box["bbox_2d"] = [scaled_x1, box_coords[1], scaled_x2, box_coords[3]]

    max_y = max(box["bbox_2d"][3] for box in bounding_boxes)
    if max_y > height:
        logger.warning(f'Max bbox y {max_y} exceeds image height {height}. Rescaling y-coordinates.')
        y_scale_factor = height / max_y
        for bounding_box in bounding_boxes:
            box_coords = bounding_box["bbox_2d"]
            scaled_y1 = int(box_coords[1] * y_scale_factor)
            scaled_y2 = int(box_coords[3] * y_scale_factor)
            bounding_box["bbox_2d"] = [box_coords[0], scaled_y1, box_coords[2], scaled_y2]

    return bounding_boxes


def resize_vlm_elements(result: Dict[str, Any], elements_keys: List[str], input_image_height: int,
                        input_image_width: int,
                        original_image_height: int, original_image_width: int,
                        version: Literal['qwen25vl', 'qwen3vl', 'qwen3.5', 'qwen3.6', 'gemini'] = "qwen25vl"):
    """
    将大模型输出字典里的每个画面元素边界框相关字段，转换为[{"bbox_2d": [x1, x2, y1, y2], ...}]格式

    input_image_height: 用户缩放后提供给vlm模型的图像高度
    input_image_width: 用户缩放后提供给vlm模型的图像宽度
    original_image_height: 原图高度
    original_image_width: 原图宽度
    """
    for bbox_key in elements_keys:
        if bbox_key not in result:
            continue
        if result[bbox_key] is None or not result[bbox_key]:
            continue

        if isinstance(result[bbox_key], dict):
            result[bbox_key] = resize_vlm_bboxes([result[bbox_key]], input_image_height, input_image_width,
                                                 version=version)
        elif isinstance(result[bbox_key], list):
            if isinstance(result[bbox_key][0], dict):
                result[bbox_key] = resize_vlm_bboxes(result[bbox_key], input_image_height, input_image_width,
                                                     version=version)
            elif isinstance(result[bbox_key][0], list):
                result[bbox_key] = resize_vlm_bboxes([{"bbox_2d": bbox} for bbox in result[bbox_key]],
                                                     input_image_height, input_image_width, version=version)
            elif isinstance(result[bbox_key][0], int):
                result[bbox_key] = resize_vlm_bboxes([{"bbox_2d": result[bbox_key]}], input_image_height,
                                                     input_image_width, version=version)

        for element in result[bbox_key]:
            element["bbox_2d"][0] = int(element["bbox_2d"][0] * original_image_width / input_image_width)
            element["bbox_2d"][1] = int(element["bbox_2d"][1] * original_image_height / input_image_height)
            element["bbox_2d"][2] = int(element["bbox_2d"][2] * original_image_width / input_image_width)
            element["bbox_2d"][3] = int(element["bbox_2d"][3] * original_image_height / input_image_height)
    return result


def image_file_to_base64(filepath: str, as_data_uri: bool = True):
    """将图像文件转为base64，能自动识别MIME类型"""
    mime_type, _ = mimetypes.guess_type(filepath)
    with open(filepath, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode('ascii')
    if as_data_uri:
        return f"data:{mime_type};base64,{encoded_str}"
    else:
        return encoded_str


def pil_image_to_base64(image: Image.Image, image_format: str = 'PNG', as_data_uri: bool = True) -> str:
    """
    将 PIL Image 对象转换为 Base64 字符串。

    :param image: PIL.Image.Image 对象
    :param image_format: 图像格式, 如 'PNG', 'JPEG'
    :param as_data_uri: 是否返回 Data URI
    :return: Base64 编码的字符串
    """
    buffered = io.BytesIO()
    image.save(buffered, format=image_format)
    img_byte = buffered.getvalue()
    base64_string = base64.b64encode(img_byte).decode('utf-8')
    if as_data_uri:
        mime_type = f"image/{image_format.lower()}"
        return f"data:{mime_type};base64,{base64_string}"
    else:
        return base64_string


def download_image(image_url: str, save_path: str, headers: dict = None):
    """
    下载图像到本地

    Args:
        image_url: 图像 URL
        save_path: 保存路径
        headers: 可选的 HTTP 请求头
    """
    try:
        response = requests.get(image_url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()  # 如果HTTP状态码不是200，则引发异常
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.debug(f"Downloaded image to {save_path}")
        return save_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        return None


def crop_image(bbox: List[int], input_image_path: str, output_image_path: str, lossless: bool = True):
    """
    裁剪图像并保存
    
    Args:
        bbox: 边界框 [x1, y1, x2, y2]
        input_image_path: 输入图像路径
        output_image_path: 输出图像路径
        lossless: 是否无损保存（默认True，使用PNG格式）
    """
    if len(bbox) == 4:
        img = Image.open(input_image_path)
        x1, y1, x2, y2 = bbox
        # 确保 bbox 在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

        # 裁剪图像
        cropped_img = img.crop((x1, y1, x2, y2))

        # 确定保存格式
        output_ext = os.path.splitext(output_image_path)[1].lower()
        
        if lossless:
            # 无损保存：使用PNG格式
            # 如果输出路径不是PNG，自动修改扩展名
            if not output_ext.endswith('.png'):
                base_path = os.path.splitext(output_image_path)[0]
                output_image_path = f"{base_path}.png"
                logger.debug(f"Changed output path to PNG for lossless saving: {output_image_path}")
            
            # 保存为PNG，保持原始质量
            cropped_img.save(output_image_path, 'PNG')
        else:
            # 如果图像是 RGBA 模式，转换为 RGB 以支持 JPEG 格式
            if cropped_img.mode == 'RGBA':
                # 创建白色背景
                rgb_img = Image.new('RGB', cropped_img.size, (255, 255, 255))
                # 将 RGBA 图像粘贴到白色背景上
                rgb_img.paste(cropped_img, mask=cropped_img.split()[3])  # 使用 alpha 通道作为 mask
                cropped_img = rgb_img

            # 保存裁剪后的图像
            cropped_img.save(output_image_path)
        
        logger.debug(f"Saved cropped product image to {output_image_path}")
        return output_image_path
    else:
        logger.error(f"Invalid bbox: {bbox}")
        return None


def compress_image(image_path: str, max_size_mb: float = 10, quality: int = 85) -> str:
    """
    压缩图像文件大小到指定限制以下
    
    Args:
        image_path: 输入图像路径
        max_size_mb: 最大文件大小（MB），默认为 10MB
        quality: 初始 JPEG 质量（1-100），默认为 85
    
    Returns:
        压缩后的图像文件路径
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # 检查原始文件大小
    original_size = os.path.getsize(image_path)
    if original_size <= max_size_bytes:
        logger.debug(f"Image size {original_size / 1024 / 1024:.2f}MB is already within limit")
        return image_path
    
    # 生成压缩后的文件路径
    base_name, ext = os.path.splitext(image_path)
    compressed_path = f"{base_name}_compressed.jpg"
    
    img = Image.open(image_path)
    
    # 如果图像是 RGBA 模式，转换为 RGB
    if img.mode == 'RGBA':
        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
        rgb_img.paste(img, mask=img.split()[3])
        img = rgb_img
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    current_quality = quality
    
    # 逐步降低质量直到满足大小限制
    while current_quality >= 10:
        img.save(compressed_path, 'JPEG', quality=current_quality, optimize=True)
        compressed_size = os.path.getsize(compressed_path)
        
        if compressed_size <= max_size_bytes:
            logger.debug(f"Compressed image from {original_size / 1024 / 1024:.2f}MB to {compressed_size / 1024 / 1024:.2f}MB (quality={current_quality})")
            return compressed_path
        
        current_quality -= 5
    
    # 如果质量降到最低仍然超过限制，尝试调整尺寸
    width, height = img.size
    scale_factor = 0.9
    
    while scale_factor >= 0.5:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_img.save(compressed_path, 'JPEG', quality=85, optimize=True)
        compressed_size = os.path.getsize(compressed_path)
        
        if compressed_size <= max_size_bytes:
            logger.debug(f"Compressed image from {original_size / 1024 / 1024:.2f}MB to {compressed_size / 1024 / 1024:.2f}MB (scale={scale_factor})")
            return compressed_path
        
        scale_factor -= 0.1
    
    # 如果所有方法都失败，使用最后一个压缩版本
    logger.warning(f"Could not compress image below {max_size_mb}MB. Final size: {compressed_size / 1024 / 1024:.2f}MB")
    return compressed_path


def get_image_extension(image_path: str) -> str:
    """
    获取图像文件的扩展名（包含点）
    如果无法确定扩展名，默认返回 .jpg
    """
    ext = os.path.splitext(image_path)[1].lower()
    # 支持的图像格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff']
    if ext in supported_formats:
        return ext
    return '.jpg'
