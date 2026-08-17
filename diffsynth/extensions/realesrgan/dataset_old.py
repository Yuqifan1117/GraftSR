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

    def _get_crop_params(self, gt_img, lq_mask_img=None, output_size=(512, 512)):
        """
        统一生成 crop_params：优先 mask-aware，否则标准随机裁剪。
        所有 crop 操作都应使用此函数返回的同一组参数。
        """
        crop_h, crop_w = output_size
        force_valid_prob = getattr(self.args, 'force_valid_crop_prob', 0.5)

        if lq_mask_img is not None:
            mask_np = np.array(lq_mask_img)
            valid_coords = np.argwhere(mask_np > 128)

            if len(valid_coords) > 0 and random.random() < force_valid_prob:
                rand_idx = random.randint(0, len(valid_coords) - 1)
                cy, cx = valid_coords[rand_idx]
                h, w = mask_np.shape
                top_min = max(0, cy - crop_h + 1)
                top_max = min(cy, h - crop_h)
                left_min = max(0, cx - crop_w + 1)
                left_max = min(cx, w - crop_w)
                if top_min <= top_max and left_min <= left_max:
                    top = random.randint(top_min, top_max)
                    left = random.randint(left_min, left_max)
                    return (top, left, crop_h, crop_w)

        # fallback: 标准随机裁剪
        return transforms.RandomCrop.get_params(gt_img, output_size=output_size)


    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):
        if self.split == 'train':
            short_edge_size = 512
            if self.use_resize_before_crop:
                if self.random_short_edge:
                    short_edge_size = random.randint(512, self.short_edge_size)
                else:
                    short_edge_size = self.short_edge_size

            if self.args.highquality_dataset_txt_paths is not None:
                if np.random.uniform() < self.args.prob:
                    gt_img, gt_img_path, actual_idx = safe_open_image(idx, self.gt_list)
                    use_ref = True
                    prompt_list = self.prompt_list
                    prompt_idx = actual_idx
                    current_lq_list = self.lq_list
                else:
                    hq_idx = random.randint(0, len(self.hq_gt_list) - 1)
                    gt_img, gt_img_path, actual_idx = safe_open_image(hq_idx, self.hq_gt_list)
                    use_ref = False
                    prompt_list = self.hq_prompt_list
                    prompt_idx = actual_idx
                    current_lq_list = getattr(self, 'hq_lq_list', [])
            else:
                gt_img, gt_img_path, actual_idx = safe_open_image(idx, self.gt_list)
                use_ref = True
                prompt_list = self.prompt_list
                prompt_idx = actual_idx
                current_lq_list = self.lq_list
            

            aligned_crop = getattr(self.args, 'aligned_crop_gt_ref', False) and use_ref
            use_real_lq = len(current_lq_list) > 0

            # ✅ 新增：提前加载 lq_mask，用于 mask-aware crop
            lq_mask_full_for_crop = None
            if len(self.hq_ori_mask_list) > 0:
                lq_mask_idx = actual_idx % len(self.lq_mask_list)
                lq_mask_path = self.lq_mask_list[lq_mask_idx]
                if os.path.exists(lq_mask_path):
                    lq_mask_full_for_crop = Image.open(lq_mask_path).convert('L')
                    if self.use_resize_before_crop:
                        lq_mask_full_for_crop = lq_mask_full_for_crop.resize(gt_img.size, Image.Resampling.NEAREST)

            # ✅ 新增：一次性确定 crop_params，后续所有分支复用
            crop_params = self._get_crop_params(gt_img, lq_mask_img=lq_mask_full_for_crop, output_size=(512, 512))


            if use_real_lq:
                # 真实 LQ 模式：尝试加载 paired LQ，失败则 fallback 到在线退化
                lq_path = current_lq_list[actual_idx] if actual_idx < len(current_lq_list) else None
                lq_img = None
                if lq_path and os.path.exists(lq_path):
                    try:
                        lq_img = Image.open(lq_path).convert('RGB')
                    except Exception:
                        print(f"Warning: Failed to open real LQ image {lq_path}, fallback to online degradation")

                if lq_img is not None:
                    # 真实 LQ 加载成功，GT 和 LQ 同步 crop
                    if self.use_resize_before_crop:
                        gt_img = resize_short_edge(gt_img, short_edge_size)
                        lq_img = resize_short_edge(lq_img, short_edge_size)

                    # 对真实 LQ 再叠加轻度退化（在 crop 前做，保证 full_lq 和 lq 块一致）
                    real_lq_degrade_prob = getattr(self.args, 'real_lq_degrade_prob', 0.0)
                    if real_lq_degrade_prob > 0 and random.random() < real_lq_degrade_prob:
                        lq_np = np.asarray(lq_img).astype(np.float32) / 255.
                        _, lq_degraded_t = self.degradation.light_degrade_process(lq_np, resize_bak=True)
                        lq_img = tensor_01_to_pil(lq_degraded_t.squeeze(0))

                    # 三条件模式：在 crop 前保存完整 LQ（已包含轻度退化）
                    full_lq_pil = None
                    if self.use_full_lq_condition:
                        full_lq_pil = self.full_lq_preproc(lq_img.copy())

                    # crop_params = transforms.RandomCrop.get_params(gt_img, output_size=(512, 512))
                    gt_img = F.crop(gt_img, *crop_params)

                    lq_w, lq_h = lq_img.size
                    crop_top, crop_left, crop_height, crop_width = crop_params
                    if lq_h >= crop_top + crop_height and lq_w >= crop_left + crop_width:
                        lq_img = F.crop(lq_img, *crop_params)
                    else:
                        lq_img = self.crop_preproc(lq_img)

                    example = {}
                    example["gt"] = gt_img
                    example["lq"] = lq_img
                    if full_lq_pil is not None:
                        example["full_lq"] = full_lq_pil
                else:
                    # LQ 打不开，fallback 到在线退化
                    use_real_lq = False

            if not use_real_lq:
                # 在线退化模式（原有逻辑）
                if self.use_resize_before_crop:
                    gt_img = resize_short_edge(gt_img, short_edge_size)

                if self.use_full_lq_condition:
                    # 三条件模式：先对完整 GT 做退化得到完整 LQ，再同步 crop GT 和 LQ
                    # 保证 lq 块是 full_lq 对应区域的 crop
                    full_gt_np = np.asarray(gt_img).astype(np.float32) / 255.
                    full_gt_t, full_lq_t = self._degrade(full_gt_np.copy(), resize_bak=True)
                    full_lq_pil = tensor_01_to_pil(full_lq_t.squeeze(0))
                    full_gt_pil = tensor_01_to_pil(full_gt_t.squeeze(0))

                    # 同步 crop GT 和 LQ
                    # crop_params = transforms.RandomCrop.get_params(full_gt_pil, output_size=(512, 512))
                    gt_img = F.crop(full_gt_pil, *crop_params)
                    lq_img = F.crop(full_lq_pil, *crop_params)

                    example = {}
                    example["gt"] = gt_img
                    example["lq"] = lq_img
                    example["full_lq"] = self.full_lq_preproc(full_lq_pil)
                else:
                    # 原有逻辑：先 crop GT，再对 crop 后的 GT 做退化
                    if aligned_crop:
                        # crop_params = transforms.RandomCrop.get_params(gt_img, output_size=(512, 512))
                        gt_img = F.crop(gt_img, *crop_params)
                    else:
                        crop_params = None
                        gt_img = self.crop_preproc(gt_img)

                    gt_np = np.asarray(gt_img) / 255.

                    # ======== 双退化模式：同一管线跑两次，按分数排序 ========
                    dual_noise_cond_degrade = getattr(self.args, 'dual_noise_cond_degrade', False)
                    if dual_noise_cond_degrade:
                        output_t_1, lq_t_1, score_1 = self.degradation.degrade_process_with_score(
                            gt_np.copy(), resize_bak=True)
                        _, lq_t_2, score_2 = self.degradation.degrade_process_with_score(
                            gt_np.copy(), resize_bak=True)
                        output_t_1 = output_t_1.squeeze(0)
                        lq_t_1, lq_t_2 = lq_t_1.squeeze(0), lq_t_2.squeeze(0)

                        # 分数低 → 轻退化 → lq（给噪声流）
                        # 分数高 → 重退化 → lq_for_cond（给条件流）
                        if score_1 <= score_2:
                            lq_light, lq_heavy = lq_t_1, lq_t_2
                        else:
                            lq_light, lq_heavy = lq_t_2, lq_t_1

                        example = {}
                        example["gt"] = tensor_01_to_pil(output_t_1)
                        example["lq"] = tensor_01_to_pil(lq_light)
                        example["lq_for_cond"] = tensor_01_to_pil(lq_heavy)
                    else:
                        # 原有逻辑：单次退化
                        output_t, img_t = self._degrade(gt_np, resize_bak=True)
                        output_t, img_t = output_t.squeeze(0), img_t.squeeze(0)

                        example = {}
                        example["gt"] = tensor_01_to_pil(output_t)
                        example["lq"] = tensor_01_to_pil(img_t)

            # 加载参考图像
            if use_ref and self.args.ref_txt_paths is not None and len(self.ref_list) > 0:
                dropout_lq = False
                if len(self.similarity_list) > 0:
                    sim_idx = actual_idx % len(self.similarity_list)
                    similarity_value = self.similarity_list[sim_idx]
                    # 根据相似度阈值决定是否丢弃LQ条件输入
                    lq_dropout_threshold = getattr(self.args, 'lq_dropout_similarity_threshold', 0.95)
                    if similarity_value > lq_dropout_threshold:
                        dropout_lq = True
                else:
                    dropout_lq = True
                example["dropout_lq"] = dropout_lq

                ref_idx = actual_idx % len(self.ref_list)
                ref_path = self.ref_list[ref_idx]
                if ref_path and os.path.exists(ref_path):
                    ref_img = Image.open(ref_path).convert('RGB')
                

                    if self.flexible_ref_resolution:
                        # flexible 模式：短边 resize + 16 对齐，不 crop
                        # ref 不应该crop，和lq空间是不对齐的
                        ref_img = self.ref_preproc(ref_img)
                    else:
                        if self.use_resize_before_crop:
                            ref_img = resize_short_edge(ref_img, short_edge_size)
                        # if crop_params is not None:
                        #     ref_w, ref_h = ref_img.size
                        #     crop_top, crop_left, crop_height, crop_width = crop_params
                        #     if ref_h >= crop_top + crop_height and ref_w >= crop_left + crop_width:
                        #         ref_img = F.crop(ref_img, *crop_params)
                        #     else:
                        #         ref_img = self.crop_preproc(ref_img)
                        # else:
                            ref_img = self.crop_preproc(ref_img)

                    ref_tensor = F.to_tensor(ref_img)
                    example["ref"] = tensor_01_to_pil(ref_tensor)

                    if self.ref_light_degrade_prob > 0 and random.random() < self.ref_light_degrade_prob:
                        ref_np = np.asarray(example["ref"]).astype(np.float32) / 255.
                        _, ref_lq_t = self.degradation.light_degrade_process(ref_np, resize_bak=True)
                        example["ref"] = tensor_01_to_pil(ref_lq_t.squeeze(0))
                else:
                    # 无有效 ref，走 dropout_ref 逻辑
                    example["dropout_ref"] = True
            elif self.args.ref_txt_paths is not None and len(self.ref_list) > 0:
                # highquality 数据集：随机抽一张错配的 ref，让模型学会不参考不匹配的 ref
                if getattr(self.args, 'hq_mismatched_ref', False):
                    mismatched_ref_idx = random.randint(0, len(self.ref_list) - 1)
                    ref_img = Image.open(self.ref_list[mismatched_ref_idx]).convert('RGB')

                    if self.flexible_ref_resolution:
                        ref_img = self.ref_preproc(ref_img)
                    else:
                        if self.use_resize_before_crop:
                            ref_img = resize_short_edge(ref_img, short_edge_size)
                        ref_img = self.crop_preproc(ref_img)

                    ref_tensor = F.to_tensor(ref_img)
                    example["ref"] = tensor_01_to_pil(ref_tensor)

                    if self.ref_light_degrade_prob > 0 and random.random() < self.ref_light_degrade_prob:
                        ref_np = np.asarray(example["ref"]).astype(np.float32) / 255.
                        _, ref_lq_t = self.degradation.light_degrade_process(ref_np, resize_bak=True)
                        example["ref"] = tensor_01_to_pil(ref_lq_t.squeeze(0))
                else:
                    # dropout ref：不设 ref，走单 condition 路径
                    example["dropout_ref"] = True
            # 加载 SAM mask（LQ mask + REF mask）
            if len(self.lq_mask_list) > 0:
                # ✅ 复用前面已加载的 mask，无需重新读取
                if lq_mask_full_for_crop is not None:
                    lq_mask_img = lq_mask_full_for_crop.copy()  # copy 避免后续 crop 污染原始数据
                else:
                    # fallback：如果前面没加载成功，再尝试加载
                    lq_mask_idx = actual_idx % len(self.lq_mask_list)
                    lq_mask_path = self.lq_mask_list[lq_mask_idx]
                    if os.path.exists(lq_mask_path):
                        lq_mask_img = Image.open(lq_mask_path).convert('L')
                        if self.use_resize_before_crop:
                            lq_mask_img = lq_mask_img.resize(gt_img.size, Image.Resampling.NEAREST)
                    else:
                        lq_mask_img = None

                if lq_mask_img is not None:
                    if crop_params is not None:
                        lq_mask_img = F.crop(lq_mask_img, *crop_params)
                    else:
                        lq_mask_img = lq_mask_img.resize((512, 512), Image.Resampling.NEAREST)
                    example["lq_mask"] = lq_mask_img

            if len(self.ref_mask_list) > 0:
                ref_mask_idx = actual_idx % len(self.ref_mask_list)
                ref_mask_path = self.ref_mask_list[ref_mask_idx]
                if os.path.exists(ref_mask_path):
                    ref_mask_img = Image.open(ref_mask_path).convert('L')
                    # ref_mask 需要 resize 到与 ref 图相同尺寸
                    ref_size = example["ref"].size  # (W, H)
                    ref_mask_img = ref_mask_img.resize(ref_size, Image.BILINEAR)
                    example["ref_mask"] = ref_mask_img


            if not self.args.use_qwen:
                is_condition_dropped = example.get("dropout_ref", False)
                random_value = random.random()
                if random_value < self.args.null_text_ratio and not is_condition_dropped:
                    text = ""
                else:
                    if prompt_list is not None:
                        text = safe_open_prompt(prompt_idx, prompt_list)
                    else:
                        text = get_prompt_given_img_path(gt_img_path)

                    if getattr(self.args, 'use_generate_prompt_prefix', False):
                        # 保留不带前缀的原始 prompt，供判别器使用
                        example['raw_text'] = text
                        if example.get("ref") and not example.get("dropout_ref", False):
                            text = GENERATE_WITH_REF_PROMPT_PREFIX + text
                        else:
                            text = GENERATE_WO_REF_PROMPT_PREFIX + text
                    if getattr(self.args, 'use_edit_prompt_prefix', False):
                        example['raw_text'] = text
                        if example.get("ref") and not example.get("dropout_ref", False):
                            text = EDIT_WITH_REF_PROMPT_PREFIX + text
                        else:
                            text = EDIT_WO_REF_PROMPT_PREFIX + text

                example['text'] = text
                if 'raw_text' not in example:
                    example['raw_text'] = text
            
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