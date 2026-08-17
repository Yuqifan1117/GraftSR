"""
copy from https://github.com/csslc/PiSA-SR/blob/main/src/datasets/dataset.py
change record: https://www.diffchecker.com/wEIJtoyR/
"""
import os
import random
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

import numpy as np
from .realesrgan import RealESRGAN_degradation

import pillow_heif
pillow_heif.options.DISABLE_SECURITY_LIMITS = True

# 固定 prompt 前缀：描述条件输入的性质，引导模型理解 condition_image 的语义
GENERATE_WO_REF_PROMPT_PREFIX = (
    "A high-resolution output image with sharp edges, fine textures. "
)
GENERATE_WITH_REF_PROMPT_PREFIX = (
    "The additional input conditions include a low-quality image that provides the spatial structure and layout, "
    "and a high-quality reference image that provides the texture details and content appearance. "
    "A high-resolution output image with sharp edges, fine textures. "
)

EDIT_WO_REF_PROMPT_PREFIX = (
    "Enhance the low-quality image to high resolution with sharp details and fine textures: "
)

EDIT_WITH_REF_PROMPT_PREFIX = (
    "Enhance the low-quality image to high resolution with sharp details and fine textures, "
    "using the high-quality reference image as a guide for texture and appearance: "
)


def get_prompt_given_img_path(img_path, img_subdir="/img/", prompt_subdir="/prompt/"):
    txt_path = img_path.replace(img_subdir, prompt_subdir)

    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif']:
        if txt_path.lower().endswith(ext):
            txt_path = txt_path[:-len(ext)] + '.txt'
            break

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()  # 读取内容并去除首尾空白字符
    except FileNotFoundError:
        print(f"Warning: Prompt file not found for {img_path}")
        return ""
    except Exception as e:
        print(f"Error reading prompt file {txt_path}: {str(e)}")
        return ""

def tensor_01_to_pil(tensor):
    tensor = tensor.detach().cpu()
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    tensor = tensor.permute(1, 2, 0)
    image = Image.fromarray(tensor.numpy())
    return image


def resize_short_edge(img, target_size):
    """
    将图像的短边缩放到目标尺寸，保持长宽比
    """
    w, h = img.size
    short_edge = min(w, h)
    if short_edge <= target_size:
        return img
    if w < h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def safe_open_image(idx, paths, min_size=512):
    """
    安全地打开图片文件，失败或尺寸不足时随机重试其他路径。

    Args:
        idx: 目标索引
        paths: 图片路径列表
        min_size: 宽和高的最小尺寸，小于此值的图片会被跳过

    Returns:
        (image, path, actual_idx): PIL Image, 实际路径, 实际索引
    """
    if 0 <= idx < len(paths) and os.path.exists(paths[idx]):
        try:
            img = Image.open(paths[idx]).convert('RGB')
            if img.size[0] >= min_size and img.size[1] >= min_size:
                return img, paths[idx], idx
        except Exception:
            pass

    max_retries = 200
    for _attempt in range(max_retries):
        random_idx = random.randint(0, len(paths) - 1)
        p = paths[random_idx]
        if os.path.exists(p):
            try:
                img = Image.open(p).convert('RGB')
                if img.size[0] >= min_size and img.size[1] >= min_size:
                    return img, p, random_idx
            except Exception:
                continue
    original_path = paths[idx] if 0 <= idx < len(paths) else f"idx={idx} out of range"
    raise RuntimeError(
        f"[safe_open_image] Failed after {max_retries} retries. "
        f"Original path: {original_path}. "
        f"Dataset may be inaccessible on this node or all images < {min_size}px. "
        f"Sample paths: {paths[:3]}"
    )


def safe_open_prompt(idx, prompt_paths):
    """从 prompt 文件列表中读取指定索引的 prompt 内容"""
    prompt_path = prompt_paths[idx]
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading prompt file {prompt_path}: {str(e)}")
        return ""

def preproc_with_pixels(img, max_pixels=1366 * 768):
    w, h = img.size
    current_pixels = w * h
    if current_pixels > max_pixels:
        scale = (max_pixels / current_pixels) ** 0.5
        new_w = max(16, int(w * scale) // 16 * 16)
        new_h = max(16, int(h * scale) // 16 * 16)
    else:
        new_w = max(16, w // 16 * 16)
        new_h = max(16, h // 16 * 16)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def load_txt_paths(txt_paths: list[str] | None, target_len: int | None = None):
    txt_list = []
    if txt_paths is not None:
        for txt_path in txt_paths:
            with open(txt_path, 'r') as f:
                txt_list.extend([line.strip() for line in f])
        if target_len is not None:
            assert len(txt_list) == target_len, f"{txt_paths} entries ({len(txt_list)}) must match target_len ({target_len})"
    return txt_list

class PairedSROnlineTxtDataset(torch.utils.data.Dataset):
    """
        args should have:
        * deg_file_path
        * dataset_txt_paths
        * use_qwen
        * highquality_dataset_txt_paths
        * null_text_ratio
    """
    def __init__(self, split=None, args=None):
        super().__init__()

        self.args = args
        self.split = split
        if split == 'train':
            self.degradation = RealESRGAN_degradation(args.deg_file_path, device='cpu')
            self.use_resize_before_crop = getattr(args, 'use_resize_before_crop', False)
            if self.use_resize_before_crop:
                self.short_edge_size = getattr(args, 'short_edge_size', 1024)
                self.random_short_edge = getattr(args, 'random_short_edge', False)

            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((512, 512)),
            ])

            # flexible ref 模式：ref 不 crop，等比缩放使总像素不超过预算 + 16 对齐（不放大）
            self.flexible_ref_resolution = getattr(args, 'flexible_ref_resolution', False)
            assert args.ref_txt_paths is not None and self.flexible_ref_resolution
            self.ref_max_pixels = getattr(args, 'flexible_ref_max_pixels', 1366 * 768)
            # 三条件模式：LQ 块 + 完整 LQ + REF
            self.use_full_lq_condition = getattr(args, 'use_full_lq_condition', False)
            if self.use_full_lq_condition:
                self.full_lq_max_pixels = getattr(args, 'full_lq_max_pixels', 1366 * 768)
            
            self.use_attn_mask = getattr(args, 'use_attn_mask', False)

            self.gt_list = load_txt_paths(args.dataset_txt_paths)
            self.prompt_list = load_txt_paths(args.main_prompt_txt_paths, target_len=len(self.gt_list))
            self.ref_list = load_txt_paths(args.ref_txt_paths, target_len=len(self.gt_list))
            self.hq_ori_mask_list = load_txt_paths(args.hq_ori_mask_txt_paths, target_len=len(self.gt_list))
            self.ref_crop_mask_list = load_txt_paths(args.ref_crop_mask_txt_paths, target_len=len(self.gt_list))
            self.hq_gt_list = load_txt_paths(args.highquality_dataset_txt_paths)
            self.hq_prompt_list = load_txt_paths(args.highquality_prompt_txt_paths, target_len=len(self.hq_gt_list))
            self.hq_prob = getattr(args, 'hq_prob', 0.0)

        elif split == 'test':
            self.input_folder = os.path.join(args.dataset_test_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(args.dataset_test_folder, "test_HR")
            self.ref_folder = os.path.join(args.dataset_test_folder, "test_ref")
            self.lr_list = []
            self.gt_list = []
            self.ref_list = []

            lr_names = os.listdir(os.path.join(self.input_folder))
            gt_names = os.listdir(os.path.join(self.output_folder))
            assert len(lr_names) == len(gt_names)
            for i in range(len(lr_names)):
                self.lr_list.append(os.path.join(self.input_folder, lr_names[i]))
                self.gt_list.append(os.path.join(self.output_folder,gt_names[i]))

            if os.path.exists(self.ref_folder):
                ref_names = os.listdir(self.ref_folder)
                for ref_name in ref_names:
                    self.ref_list.append(os.path.join(self.ref_folder, ref_name))
                assert len(self.ref_list) == len(self.gt_list)

            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop((args.resolution_ori, args.resolution_ori)),
                transforms.Resize((args.resolution_tgt, args.resolution_tgt)),
            ])
            assert len(self.lr_list) == len(self.gt_list)

    def _degrade(self, img_np, resize_bak=True):
        """根据概率选择图像退化或视频截图退化"""
        if np.random.uniform() < self.video_screenshot_degrade_prob:
            return self.degradation.video_screenshot_degrade_process(img_np, resize_bak=resize_bak)
        return self.degradation.degrade_process(img_np, resize_bak=resize_bak)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            example = {}
            if len(self.hq_gt_list) > 0 and np.random.uniform() < self.hq_prob:
                hq_idx = random.randint(0, len(self.hq_gt_list) - 1)
                gt_img, gt_img_path, actual_idx = safe_open_image(hq_idx, self.hq_gt_list)
                use_hq = True
                prompt_list = self.hq_prompt_list
            else:
                gt_img, gt_img_path, actual_idx = safe_open_image(idx, self.gt_list)
                use_hq = False
                prompt_list = self.prompt_list

            if self.use_resize_before_crop:
                if self.random_short_edge:
                    short_edge_size = random.randint(1024, self.short_edge_size)
                else:
                    short_edge_size = self.short_edge_size
                gt_img = resize_short_edge(gt_img, short_edge_size)


            full_gt_t, full_lq_t = self.degradation.degrade_process(np.asarray(gt_img) / 255., resize_bak=True)
            # 将全图 GT 和 LQ 转为 PIL
            full_gt_pil = tensor_01_to_pil(full_gt_t.squeeze(0))
            full_lq_pil = tensor_01_to_pil(full_lq_t.squeeze(0))

            # 用相同的 crop 参数裁剪 GT 和 LQ（在缩小前 crop，保证尺寸一致）

            if len(self.hq_ori_mask_list) > 0 and len(self.ref_crop_mask_list) > 0:
                
                lq_mask_full = Image.open(self.hq_ori_mask_list[actual_idx]).convert('L')
                if self.use_resize_before_crop:
                    lq_mask_full = lq_mask_full.resize(gt_img.size, Image.Resampling.NEAREST)
                
                mask_np = np.array(lq_mask_full)
                valid_coords = np.argwhere(mask_np > 128)  # (H, W) 格式的有效像素坐标
                
                # 以 prob 概率强制裁到有效区域，否则正常随机裁剪
                force_valid_prob = getattr(self.args, 'force_valid_crop_prob', 0.0)
                crop_h, crop_w = 512, 512
                
                if len(valid_coords) > 0 and random.random() < force_valid_prob:
                    # 随机选一个有效像素作为 crop 中心附近
                    rand_idx = random.randint(0, len(valid_coords) - 1)
                    cy, cx = valid_coords[rand_idx]
                    # 计算合法的 crop 左上角范围
                    top_min = max(0, cy - crop_h + 1)
                    top_max = min(cy, mask_np.shape[0] - crop_h)
                    left_min = max(0, cx - crop_w + 1)
                    left_max = min(cx, mask_np.shape[1] - crop_w)
                    if top_min <= top_max and left_min <= left_max:
                        top = random.randint(top_min, top_max)
                        left = random.randint(left_min, left_max)
                        crop_params = (top, left, crop_h, crop_w)
                    else:
                        # fallback: 有效区域太靠边，退化为普通随机裁剪
                        crop_params = transforms.RandomCrop.get_params(full_gt_pil, output_size=(crop_h, crop_w))
                else:
                    crop_params = transforms.RandomCrop.get_params(full_gt_pil, output_size=(crop_h, crop_w))
                
                lq_crop_mask = F.crop(lq_mask_full, *crop_params)
                example["lq_mask"] = lq_crop_mask
                ref_mask = Image.open(self.ref_crop_mask_list[actual_idx]).convert('L')
                ref_mask = preproc_with_pixels(ref_mask, self.ref_max_pixels)
                example["ref_mask"] = ref_mask
            else:
                crop_params = transforms.RandomCrop.get_params(full_gt_pil, output_size=(512, 512))
            

            gt_crop_pil = F.crop(full_gt_pil, *crop_params)
            lq_crop_pil = F.crop(full_lq_pil, *crop_params)
                
            

            if self.use_full_lq_condition:
                example["full_lq"] = preproc_with_pixels(full_lq_pil, self.full_lq_max_pixels)
            example["gt"] = gt_crop_pil
            example["lq"] = lq_crop_pil

            # 加载参考图像
            ref_dropout_prob = getattr(self.args, 'ref_dropout_prob', 0.0)
            if random.random() >= ref_dropout_prob:
                if use_hq:
                    # highquality 数据集：随机抽一张错配的 ref，让模型学会不参考不匹配的 ref
                    mismatched_ref_idx = random.randint(0, len(self.ref_list) - 1)
                    ref_img = Image.open(self.ref_list[mismatched_ref_idx]).convert('RGB')
                else:
                    ref_img = Image.open(self.ref_list[actual_idx]).convert('RGB')
                ref_img = preproc_with_pixels(ref_img, self.ref_max_pixels)
                example["ref"] = ref_img

            if random.random() < self.args.null_text_ratio:
                text = ""
                example['raw_text'] = text
            else:
                text = safe_open_prompt(actual_idx, prompt_list)
                example['raw_text'] = text

                if getattr(self.args, 'use_edit_prompt_prefix', False):
                    text = EDIT_WITH_REF_PROMPT_PREFIX + text
            example['text'] = text
                    
            return example
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')
            img_t = self.crop_preproc(input_img)
            output_t = self.crop_preproc(output_img)
            # input images scaled to -1, 1
            img_t = F.to_tensor(img_t)
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            example = {}
            example["neg_prompt"] = self.args.neg_prompt_csd
            example["null_prompt"] = ""
            example["output_pixel_values"] = output_t
            example["conditioning_pixel_values"] = img_t
            example["base_name"] = os.path.basename(self.lr_list[idx])

            if len(self.ref_list) > 0:
                ref_img = Image.open(self.ref_list[idx]).convert('RGB')
                ref_img = self.crop_preproc(ref_img)
                ref_tensor = F.to_tensor(ref_img)
                ref_tensor = F.normalize(ref_tensor, mean=[0.5], std=[0.5])
                example["ref"] = ref_tensor

            return example