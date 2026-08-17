"""
基于原生 Qwen-Image-Edit 的单步推理脚本。

与原版 test_sr.py 的核心区别：
- 使用 GeneratorEdit（原生 model_fn + 标准 LoRA + zero_cond_t）
- 不使用 tiled 分块推理（原生 edit_latents 不需要）
- 条件图直接通过 edit_latents 注入，由 zero_cond_t 自动区分条件/噪声流
- 支持单图推理（infer_one_step）和多步推理（infer）两种模式
"""
import os
os.system("sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /opt/conda/envs/python3.10.13/lib/python3.10/site-packages/basicsr/data/degradations.py")
os.system("sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /root/.local/lib/python3.10/site-packages/basicsr/data/degradations.py")
print("*-"*60)
os.system("python setup.py develop")
import argparse
import time
import shutil
import tempfile
import torch
from PIL import Image
import numpy as np

# 放宽 PIL 像素数限制，避免大图推理时触发 DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

from generator_edit import GeneratorEdit
from wavelet_color_fix import adain_color_fix, wavelet_color_fix
from metrics import MetricsAccumulator


def read_paths_from_txt(txt_path):
    """从 txt 文件中读取路径列表"""
    with open(txt_path, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    return paths


OSS_MOUNT_PREFIX = "/data/oss_bucket_0"


def _save_image(image, target_path):
    """OSS 挂载路径先写临时文件再复制，否则直接保存。"""
    if target_path.startswith(OSS_MOUNT_PREFIX):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            tmp_path = tmp_f.name
        image.save(tmp_path)
        shutil.copy2(tmp_path, target_path)
        os.remove(tmp_path)
    else:
        image.save(target_path)


def postprocess_and_save(res_img, upsampled_lq, target_w, target_h,
                         filename, output_dir, viz_dir,
                         align_method='adain', gt_path=None, ref_path=None,
                         lq_mask_path=None, ref_mask_path=None):
    """后处理：裁剪到目标尺寸、颜色对齐、保存结果和可视化对比图"""
    # 裁剪到目标尺寸（推理输出可能因 16 对齐而略大）
    cropped = res_img.crop((0, 0, target_w, target_h))

    # 颜色对齐
    if align_method == 'adain':
        output_pil = adain_color_fix(target=cropped, source=upsampled_lq)
    elif align_method == 'wavelet':
        output_pil = wavelet_color_fix(target=cropped, source=upsampled_lq)
    else:
        output_pil = cropped

    _save_image(output_pil, os.path.join(output_dir, f"{filename}.png"))

    # 可视化对比：LQ | Output | GT(如有) | REF(如有)
    viz_imgs = [upsampled_lq, output_pil]
    if gt_path:
        gt_img = Image.open(gt_path).convert('RGB')
        gt_img = gt_img.resize((target_w, target_h), Image.BICUBIC)
        viz_imgs.append(gt_img)
    if ref_path:
        ref_display = Image.open(ref_path).convert('RGB')
        ref_dw, ref_dh = ref_display.size
        scale_ratio = target_h / ref_dh
        ref_display = ref_display.resize(
            (max(1, round(ref_dw * scale_ratio)), target_h), Image.BICUBIC
        )
        viz_imgs.append(ref_display)

    # ===== 新增：将 mask 加入 viz 对比图 =====
    if lq_mask_path and os.path.exists(lq_mask_path):
        lq_mask_display = Image.open(lq_mask_path).convert('RGB')
        lq_mask_display = lq_mask_display.resize((target_w, target_h), Image.BILINEAR)
        viz_imgs.append(lq_mask_display)
    if ref_mask_path and os.path.exists(ref_mask_path):
        ref_mask_display = Image.open(ref_mask_path).convert('RGB')
        ref_mask_display = ref_mask_display.resize((target_w, target_h), Image.BILINEAR)
        viz_imgs.append(ref_mask_display)
    # ===== 新增结束 =====

    canvas_h = max(im.height for im in viz_imgs)
    canvas_w = sum(im.width for im in viz_imgs)
    canvas = Image.new('RGB', (canvas_w, canvas_h))
    x_offset = 0
    for im in viz_imgs:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width
    _save_image(canvas, os.path.join(viz_dir, f"{filename}_compare.png"))

    return output_pil


def align_to_multiple(value, multiple=16):
    """将值对齐到 multiple 的倍数"""
    return max(multiple, (value + multiple - 1) // multiple * multiple)


def pad_to_tile(size, tile_px, stride_px):
    """确保尺寸 >= tile_px 且 (尺寸 - tile_px) 能被 stride_px 整除"""
    if size <= tile_px:
        return tile_px
    return ((size - tile_px + stride_px - 1) // stride_px) * stride_px + tile_px


def apply_pixel_budget(pil_image, max_pixels, align_multiple=16):
    """
    对图片施加像素预算限制：超预算时等比缩放 + 对齐到指定倍数。

    Args:
        pil_image: PIL Image
        max_pixels: 最大总像素数预算（w * h），None 表示不限制
        align_multiple: 尺寸对齐的倍数（默认 16）

    Returns:
        处理后的 PIL Image（可能是原图引用，也可能是缩放后的新图）
    """
    if max_pixels is None:
        width, height = pil_image.size
        aligned_w = align_to_multiple(width, align_multiple)
        aligned_h = align_to_multiple(height, align_multiple)
        if (aligned_w, aligned_h) != pil_image.size:
            return pil_image.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)
        return pil_image

    width, height = pil_image.size
    current_pixels = width * height
    if current_pixels > max_pixels:
        scale_factor = (max_pixels / current_pixels) ** 0.5
        width = int(width * scale_factor)
        height = int(height * scale_factor)

    aligned_w = align_to_multiple(width, align_multiple)
    aligned_h = align_to_multiple(height, align_multiple)
    if (aligned_w, aligned_h) != pil_image.size:
        return pil_image.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)
    return pil_image


def compute_target_size(lq_w, lq_h, scale=None, target_pixels=None):
    """
    根据 scale 或 target_pixels 计算输出目标尺寸（二选一）。

    Args:
        lq_w: LQ 图像宽度
        lq_h: LQ 图像高度
        scale: 超分倍率（与 target_pixels 互斥）
        target_pixels: 输出最小像素预算（与 scale 互斥），
                       仅在 LQ 像素数不足时等比放大使 w * h ≈ target_pixels；
                       若 LQ 已满足预算则保持原尺寸。
    Returns:
        (target_w, target_h)
    """
    if target_pixels is not None:
        current_pixels = lq_w * lq_h
        if current_pixels >= target_pixels:
            target_w = lq_w
            target_h = lq_h
        else:
            ratio = (target_pixels / current_pixels) ** 0.5
            target_w = max(16, round(lq_w * ratio))
            target_h = max(16, round(lq_h * ratio))
    else:
        target_w = round(lq_w * scale)
        target_h = round(lq_h * scale)
    return target_w, target_h


def init_model(trained_ckpt, gen_start_point=750, lora_rank=128,
               lora_target_modules="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
               qwen_path=None):
    """
    初始化 GeneratorEdit 模型。

    Args:
        trained_ckpt: 训练好的 checkpoint 路径
        gen_start_point: 去噪起始时间步
        lora_rank: LoRA 秩
        lora_target_modules: LoRA 目标模块
        qwen_path: Qwen 预训练模型路径（默认从环境变量 qwen_path 读取）

    Returns:
        初始化完成的 GeneratorEdit 模型（已移到 cuda）
    """
    pretrained_qwen_path = qwen_path or os.environ["qwen_path"]

    sd_safe_tensor_path_json_format = f'''[
        [
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00001-of-00005.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00002-of-00005.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00003-of-00005.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00004-of-00005.safetensors",
            "{pretrained_qwen_path}/transformer/diffusion_pytorch_model-00005-of-00005.safetensors"
        ],
        [
            "{pretrained_qwen_path}/text_encoder/model-00001-of-00004.safetensors",
            "{pretrained_qwen_path}/text_encoder/model-00002-of-00004.safetensors",
            "{pretrained_qwen_path}/text_encoder/model-00003-of-00004.safetensors",
            "{pretrained_qwen_path}/text_encoder/model-00004-of-00004.safetensors"
        ],
        "{pretrained_qwen_path}/vae/diffusion_pytorch_model.safetensors"
    ]'''

    processor_path = f"{pretrained_qwen_path}/processor"
    if not os.path.exists(processor_path):
        processor_path = None

    model = GeneratorEdit(
        torch_dtype=torch.bfloat16,
        pretrained_weights=sd_safe_tensor_path_json_format,
        tokenizer_path=f"{pretrained_qwen_path}/tokenizer",
        processor_path=processor_path,
        learning_rate=0,
        use_gradient_checkpointing=False,
        pretrained_ckpt_path_gen=trained_ckpt,
        gen_start_point=gen_start_point,
        lora_rank=lora_rank,
        lora_target_modules=lora_target_modules,
        zero_cond_t=True,
    )

    model.pipe.requires_grad_(False)
    model = model.to(device='cuda')
    model.device = next(model.parameters()).device
    model.pipe.device = model.device
    return model


@torch.no_grad()
def infer_single(model, lq_pil, prompt="High quality, hyper detailed photo",
                 ref_pil=None, scale=None, target_pixels=None,
                 input_min_short_edge=None, tiled=False,
                 tile_size=64, tile_stride=48, tile_prompt_mode="global",
                 cfg_scale=1.0, fidelity=1.0, fidelity_mask=None,
                 use_prompt_prefix=False, use_full_lq_condition=False,
                 full_lq_max_pixels=None, ref_max_pixels=None,
                 drop_lq_crop_condition=False,
                 align_method='adain',
                 mode='one_step', num_inference_steps=30, seed=42,
                 lq_mask_pil=None, ref_mask_pil=None,
                 mask_low_noise_ratio=0.1,
                 inject_mask_latent=True):
    """
    对单张 LQ 图进行超分推理，返回增强后的 PIL Image。

    Args:
        model: 已初始化的 GeneratorEdit 模型
        lq_pil: LQ 输入图（PIL Image）
        prompt: 文本 prompt
        ref_pil: 参考图（PIL Image，可选）
        scale: 超分倍率（与 target_pixels 互斥）
        target_pixels: 输出目标像素预算（与 scale 互斥）
        input_min_short_edge: LQ 短边最小值
        tiled: 是否启用分块推理
        tile_size: latent 空间 tile 大小
        tile_stride: latent 空间 tile 步长
        tile_prompt_mode: prompt 编码模式 ('global' / 'tile')
        cfg_scale: CFG 强度
        fidelity: 保真度（标量，作为 mask=0 区域的 fidelity 值）
        fidelity_mask: 逐像素 fidelity 控制 mask（PIL Image, grayscale, 可选）。
                       mask=0(黑)使用 fidelity 参数值，mask=255(白)使用 fidelity=1.0(保持原始)，
                       中间值线性插值。仅 one_step 模式生效。
        use_prompt_prefix: 是否拼接 prompt 前缀
        use_full_lq_condition: 是否启用LQ整图条件
        full_lq_max_pixels: full_lq 条件图最大像素数
        ref_max_pixels: Ref 图最大像素数
        drop_lq_crop_condition: 是否丢弃 lq_crop 条件
        align_method: 颜色对齐方法 ('adain' / 'wavelet' / 'none')
        mode: 推理模式 ('one_step' / 'multi_step')
        num_inference_steps: 多步推理去噪步数
        seed: 多步推理随机种子

    Returns:
        增强后的 PIL Image（已裁剪到目标尺寸并做颜色对齐）
    """
    # 默认 scale
    if scale is None and target_pixels is None:
        scale = 1.0

    lq_w, lq_h = lq_pil.size

    # 短边最小值放大
    if input_min_short_edge and target_pixels is None:
        short_edge = min(lq_w, lq_h)
        if short_edge < input_min_short_edge:
            scale_factor = input_min_short_edge / short_edge
            lq_w = round(lq_w * scale_factor)
            lq_h = round(lq_h * scale_factor)

    target_w, target_h = compute_target_size(lq_w, lq_h, scale=scale, target_pixels=target_pixels)
    upsampled_lq = lq_pil.resize((target_w, target_h), Image.BICUBIC)

    # 对齐到合适的倍数
    if tiled:
        tile_px = tile_size * 8
        stride_px = tile_stride * 8
        aligned_w = pad_to_tile(target_w, tile_px, stride_px)
        aligned_h = pad_to_tile(target_h, tile_px, stride_px)
    else:
        aligned_w = align_to_multiple(target_w, 16)
        aligned_h = align_to_multiple(target_h, 16)

    if aligned_w != target_w or aligned_h != target_h:
        padded_lq = Image.new('RGB', (aligned_w, aligned_h), color=0)
        padded_lq.paste(upsampled_lq, (0, 0))
    else:
        padded_lq = upsampled_lq

    # 处理 Ref 图像素预算
    if ref_pil is not None and ref_max_pixels is not None:
        ref_pil = apply_pixel_budget(ref_pil, max_pixels=ref_max_pixels)

    # 拼接 prompt 前缀
    if use_prompt_prefix:
        if ref_pil is not None:
            prompt = (
                "Enhance the low-quality image to high resolution with sharp details and fine textures, "
                "using the high-quality reference image as a guide for texture and appearance: "
            ) + prompt
        else:
            prompt = (
                "Enhance the low-quality image to high resolution with sharp details and fine textures: "
            ) + prompt

    # 准备 full_lq 条件图
    full_lq_pil = None
    if use_full_lq_condition:
        full_lq_pil = apply_pixel_budget(padded_lq, max_pixels=full_lq_max_pixels)

    # 推理
    if mode == 'one_step':
        # 处理 fidelity_mask：与 padded_lq 构造方式一致（先 resize 到 target，再 pad 到 aligned）
        resized_fidelity_mask = None
        if fidelity_mask is not None:
            mask_at_target = fidelity_mask.convert('L').resize(
                (target_w, target_h), Image.BILINEAR
            )
            if aligned_w != target_w or aligned_h != target_h:
                resized_fidelity_mask = Image.new('L', (aligned_w, aligned_h), 0)
                resized_fidelity_mask.paste(mask_at_target, (0, 0))
            else:
                resized_fidelity_mask = mask_at_target

        # # 处理 SAM mask：与 padded_lq 构造方式一致（先 resize 到 target，再 pad 到 aligned）
        # resized_lq_mask = None
        # if lq_mask_pil is not None:
        #     lq_mask_at_target = lq_mask_pil.convert('L').resize(
        #         (target_w, target_h), Image.BILINEAR
        #     )
        #     if aligned_w != target_w or aligned_h != target_h:
        #         resized_lq_mask = Image.new('L', (aligned_w, aligned_h), 0)
        #         resized_lq_mask.paste(lq_mask_at_target, (0, 0))
        #     else:
        #         resized_lq_mask = lq_mask_at_target

        # ===== 新增：LQ mask 膨胀 + 软化 + 二值化=====
        processed_lq_mask = None

        if lq_mask_pil is not None:
            import cv2
            from PIL import ImageFilter
            dilate_radius = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_radius + 1, 2 * dilate_radius + 1))
            mask_np = cv2.dilate(np.array(lq_mask_pil), kernel, iterations=1)
            processed_lq_mask = Image.fromarray(mask_np).filter(ImageFilter.GaussianBlur(radius=1))

            mask_arr = np.array(processed_lq_mask)
            mask_arr = (mask_arr > 0).astype(np.uint8) * 255
            processed_lq_mask = Image.fromarray(mask_arr, 'L')
            
            lq_mask_pil = processed_lq_mask

        # ===== 新增结束 =====

        # ===== 新增：Ref mask 膨胀 + 软化 + 二值化=====
        processed_ref_mask = None
        if ref_mask_pil is not None:
            import cv2
            from PIL import ImageFilter
            ref_dilate_radius = 7
            ref_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ref_dilate_radius + 1, 2 * ref_dilate_radius + 1))
            ref_mask_np = cv2.dilate(np.array(ref_mask_pil), ref_kernel, iterations=1)
            processed_ref_mask = Image.fromarray(ref_mask_np).filter(ImageFilter.GaussianBlur(radius=1))

            ref_mask_arr = np.array(processed_ref_mask)
            ref_mask_arr = (ref_mask_arr > 0).astype(np.uint8) * 255
            processed_ref_mask = Image.fromarray(ref_mask_arr, 'L')

            if ref_pil is not None and ref_max_pixels is not None:
                # 先对齐到 ref_pil 当前尺寸（apply_pixel_budget 后的尺寸）
                if processed_ref_mask.size != ref_pil.size:
                    processed_ref_mask = processed_ref_mask.resize(ref_pil.size, Image.Resampling.NEAREST)
            ref_mask_pil = processed_ref_mask
        
        res_img = model.infer_one_step(
            prompt=prompt,
            negative_prompt="",
            lq_pil=padded_lq,
            ref_pil=ref_pil,
            full_lq_pil=full_lq_pil,
            cfg_scale=cfg_scale,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            tile_prompt_mode=tile_prompt_mode,
            fidelity=fidelity,
            fidelity_mask=resized_fidelity_mask,
            drop_lq_crop_condition=drop_lq_crop_condition,
            lq_mask_pil=lq_mask_pil,
            ref_mask_pil=ref_mask_pil,
            mask_low_noise_ratio=mask_low_noise_ratio,
            inject_mask_latent=inject_mask_latent,
        )
    else:
        edit_image = [padded_lq, ref_pil] if ref_pil is not None else padded_lq
        res_img = model.infer(
            prompt=prompt,
            negative_prompt="",
            edit_image=edit_image,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=aligned_h,
            width=aligned_w,
        )

    # 裁剪到目标尺寸 + 颜色对齐
    cropped = res_img.crop((0, 0, target_w, target_h))
    if align_method == 'adain':
        output_pil = adain_color_fix(target=cropped, source=upsampled_lq)
    elif align_method == 'wavelet':
        output_pil = wavelet_color_fix(target=cropped, source=upsampled_lq)
    else:
        output_pil = cropped

    return output_pil, processed_ref_mask, processed_lq_mask


@torch.no_grad()
def test(args):
    model = init_model(
        trained_ckpt=args.trained_ckpt,
        gen_start_point=args.gen_start_point,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
    )

    # 读取图片路径
    image_paths = read_paths_from_txt(args.lq_input_txt)
    total_image_count = len(image_paths)
    print(f"Total input images: {total_image_count}")

    range_parts = args.start_end.split(',')
    start_idx = int(range_parts[0]) if range_parts[0] else 0
    end_idx = int(range_parts[1]) if len(range_parts) > 1 and range_parts[1] else None
    if end_idx is not None:
        image_paths = image_paths[start_idx:end_idx]
    else:
        image_paths = image_paths[start_idx:]
    # warmup: 复制第一张到列表头部
    image_paths.insert(0, image_paths[0])

    ref_image_paths = []
    if args.ref_input_txt:
        ref_image_paths = read_paths_from_txt(args.ref_input_txt)
        assert len(ref_image_paths) == total_image_count, f"ref count {len(ref_image_paths)} must equal input count {total_image_count}"
        if end_idx is not None:
            ref_image_paths = ref_image_paths[start_idx:end_idx]
        else:
            ref_image_paths = ref_image_paths[start_idx:]
        ref_image_paths.insert(0, ref_image_paths[0])

    gt_image_paths = []
    if args.gt_input_txt:
        gt_image_paths = read_paths_from_txt(args.gt_input_txt)
        assert len(gt_image_paths) == total_image_count, f"gt count {len(gt_image_paths)} must equal input count {total_image_count}"
        if end_idx is not None:
            gt_image_paths = gt_image_paths[start_idx:end_idx]
        else:
            gt_image_paths = gt_image_paths[start_idx:]
        gt_image_paths.insert(0, gt_image_paths[0])

    prompt_paths = []
    if args.prompt_input_txt:
        prompt_paths = read_paths_from_txt(args.prompt_input_txt)
        assert len(prompt_paths) == total_image_count, f"prompt count {len(prompt_paths)} must equal input count {total_image_count}"
        if end_idx is not None:
            prompt_paths = prompt_paths[start_idx:end_idx]
        else:
            prompt_paths = prompt_paths[start_idx:]
        prompt_paths.insert(0, prompt_paths[0])

    fidelity_mask_paths = []
    if args.fidelity_mask_input_txt:
        fidelity_mask_paths = read_paths_from_txt(args.fidelity_mask_input_txt)
        assert len(fidelity_mask_paths) == total_image_count, \
            f"fidelity_mask count {len(fidelity_mask_paths)} must equal input count {total_image_count}"
        if end_idx is not None:
            fidelity_mask_paths = fidelity_mask_paths[start_idx:end_idx]
        else:
            fidelity_mask_paths = fidelity_mask_paths[start_idx:]
        fidelity_mask_paths.insert(0, fidelity_mask_paths[0])

    # 加载 LQ mask 路径
    lq_mask_paths = []
    if args.hq_ori_mask_txt_paths:
        lq_mask_paths = read_paths_from_txt(args.hq_ori_mask_txt_paths)
        assert len(lq_mask_paths) == total_image_count, \
            f"lq_mask count {len(lq_mask_paths)} must equal input count {total_image_count}"
        if end_idx is not None:
            lq_mask_paths = lq_mask_paths[start_idx:end_idx]
        else:
            lq_mask_paths = lq_mask_paths[start_idx:]
        lq_mask_paths.insert(0, lq_mask_paths[0])

    # 加载 REF mask 路径
    ref_mask_paths = []
    if args.ref_crop_mask_txt_paths:
        ref_mask_paths = read_paths_from_txt(args.ref_crop_mask_txt_paths)
        assert len(ref_mask_paths) == total_image_count, \
            f"ref_mask count {len(ref_mask_paths)} must equal input count {total_image_count}"
        if end_idx is not None:
            ref_mask_paths = ref_mask_paths[start_idx:end_idx]
        else:
            ref_mask_paths = ref_mask_paths[start_idx:]
        ref_mask_paths.insert(0, ref_mask_paths[0])
    
    # ===== 新增：打乱 ref 以提供错配 ref（在所有路径加载完成后执行）=====
    if ref_image_paths:
        import random
        random.seed(args.seed)  # 可复现
        warmup_ref = ref_image_paths[0]
        actual_refs = ref_image_paths[1:]

        if ref_mask_paths:
            warmup_ref_mask = ref_mask_paths[0]
            actual_ref_masks = ref_mask_paths[1:]
            combined = list(zip(actual_refs, actual_ref_masks))
            random.shuffle(combined)
            shuffled_refs, shuffled_ref_masks = zip(*combined)
            ref_image_paths = [warmup_ref] + list(shuffled_refs)
            ref_mask_paths = [warmup_ref_mask] + list(shuffled_ref_masks)
        else:
            random.shuffle(actual_refs)
            ref_image_paths = [warmup_ref] + actual_refs

        print(f"[INFO] REF images shuffled for mismatched testing (seed={args.seed})")
    # ===== 新增结束 =====

    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    # 初始化评测指标
    metrics_accumulator = None
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else []
    if metric_names:
        metrics_accumulator = MetricsAccumulator(
            metrics=metric_names,
            device="cuda",
            crop_border=args.crop_border,
        )
        print(f"Evaluation metrics enabled: {metric_names}, crop_border={args.crop_border}")

    total_images = len(image_paths)
    global_start = time.perf_counter()

    for idx, img_path in enumerate(image_paths):
        if not img_path or not os.path.exists(img_path):
            print(f"[{idx + 1}/{total_images}] {filename} | SKIPPED (input not found: {img_path})")
            continue
        iter_start = time.perf_counter()
        filename = os.path.splitext(os.path.basename(img_path))[0]

        # 跳过已保存有输出图像的测试（warmup 除外）
        output_path = os.path.join(args.output_dir, f"{filename}.png")
        if idx > 0 and os.path.exists(output_path):
            print(f"[{idx + 1}/{total_images}] {filename} | SKIPPED (output already exists)")
            continue

        # 加载 LQ
        lq_pil = Image.open(img_path).convert('RGB')

        # 加载 REF（如果有）
        ref_pil = None
        if ref_image_paths:
            ref_path_current = ref_image_paths[idx]
            if not os.path.exists(ref_path_current):
                print(f"[{idx + 1}/{total_images}] {filename} | SKIPPED (ref not found: {ref_path_current})")
                continue
            ref_pil = Image.open(ref_path_current).convert('RGB')

        # 读取 prompt
        prompt = "High quality, hyper detailed photo"
        if prompt_paths:
            prompt_path = prompt_paths[idx]
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    prompt = content

        # 加载 fidelity mask（如果有）
        fidelity_mask_pil = None
        if fidelity_mask_paths:
            mask_path_current = fidelity_mask_paths[idx]
            if os.path.exists(mask_path_current):
                print(f"[{idx + 1}/{total_images}] {filename} | fidelity mask loaded: {mask_path_current}")
                fidelity_mask_pil = Image.open(mask_path_current).convert('L')

        # 加载 SAM mask（如果有）
        lq_mask_pil = None
        if lq_mask_paths:
            lq_mask_path_current = lq_mask_paths[idx]
            if os.path.exists(lq_mask_path_current):
                lq_mask_pil = Image.open(lq_mask_path_current).convert('L')

        ref_mask_pil = None
        if ref_mask_paths:
            ref_mask_path_current = ref_mask_paths[idx]
            if os.path.exists(ref_mask_path_current):
                ref_mask_pil = Image.open(ref_mask_path_current).convert('L')

        # 调用 infer_single 完成推理 + 后处理
        output_pil, processed_ref_mask, processed_lq_mask = infer_single(
            model=model,
            lq_pil=lq_pil,
            prompt=prompt,
            ref_pil=ref_pil,
            scale=args.scale,
            target_pixels=args.target_pixels,
            input_min_short_edge=args.input_min_short_edge,
            tiled=args.tiled,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            tile_prompt_mode=args.tile_prompt_mode,
            cfg_scale=args.cfg,
            fidelity=args.fidelity,
            fidelity_mask=fidelity_mask_pil,
            use_prompt_prefix=args.use_prompt_prefix,
            use_full_lq_condition=args.use_full_lq_condition,
            full_lq_max_pixels=args.full_lq_max_pixels,
            ref_max_pixels=args.ref_max_pixels,
            drop_lq_crop_condition=args.drop_lq_crop_condition,
            align_method=args.align_method,
            mode=args.mode,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            lq_mask_pil=lq_mask_pil,
            ref_mask_pil=ref_mask_pil,
            mask_low_noise_ratio=args.mask_low_noise_ratio,
        )

        if idx == 0:
            iter_elapsed = time.perf_counter() - iter_start
            print(f"[WARMUP] {filename} | {iter_elapsed:.1f}s")
            continue

        # ===== 新增：如果使用了 target_pixels 推理，resize 回原始 LQ 尺寸 =====
        original_lq_w, original_lq_h = lq_pil.size
        use_original_size = False
        if use_original_size and args.target_pixels is not None and output_pil.size != (original_lq_w, original_lq_h):
            print(f"  Resize output from {output_pil.size} back to LQ size ({original_lq_w}x{original_lq_h})")
            output_pil = output_pil.resize((original_lq_w, original_lq_h), Image.BICUBIC)

        # 保存结果 + 可视化对比图（infer_single 已完成裁剪和颜色对齐）
        target_w, target_h = output_pil.size
        upsampled_lq = lq_pil.resize((target_w, target_h), Image.BICUBIC)

        # 保存膨胀后的 mask
        # mask_dir = os.path.join(args.output_dir, "masks")
        # os.makedirs(mask_dir, exist_ok=True)
        # if processed_lq_mask is not None:
        #     _save_image(processed_lq_mask, os.path.join(mask_dir, f"{filename}_lq_mask.png"))
        # if processed_ref_mask is not None:
        #     _save_image(processed_ref_mask, os.path.join(mask_dir, f"{filename}_ref_mask.png"))

        postprocess_and_save(
            res_img=output_pil,
            upsampled_lq=upsampled_lq,
            target_w=target_w,
            target_h=target_h,
            filename=filename,
            output_dir=args.output_dir,
            viz_dir=viz_dir,
            align_method='none',  # infer_single 已完成颜色对齐
            gt_path=gt_image_paths[idx] if gt_image_paths else None,
            ref_path=ref_image_paths[idx] if ref_image_paths else None,
            # lq_mask_path=os.path.join(mask_dir, f"{filename}_lq_mask.png") if processed_lq_mask else None,
            # ref_mask_path=os.path.join(mask_dir, f"{filename}_ref_mask.png") if processed_ref_mask else None,
        )

        # 评测指标（跳过第 0 张 warmup）
        if metrics_accumulator is not None and idx > 0:
            gt_img = None
            if gt_image_paths:
                gt_img = Image.open(gt_image_paths[idx]).convert("RGB")
                gt_img = gt_img.resize((target_w, target_h), Image.BICUBIC)
            per_image_metrics = metrics_accumulator.update(output_pil, gt_img, image_name=filename)
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in per_image_metrics.items() if k != "image_name")
            print(f"  Metrics: {metrics_str}")

        # 进度
        iter_elapsed = time.perf_counter() - iter_start
        total_elapsed = time.perf_counter() - global_start
        finished = idx + 1
        remaining = total_images - finished
        avg_time = total_elapsed / finished
        eta = avg_time * remaining
        print(f"[{finished}/{total_images}] {filename} | {iter_elapsed:.1f}s | avg {avg_time:.1f}s | ETA {eta:.0f}s")

    total_elapsed = time.perf_counter() - global_start
    print(f"Done. {total_images} images in {total_elapsed:.1f}s (avg {total_elapsed / total_images:.1f}s/img)")

    # 汇总指标
    if metrics_accumulator is not None:
        if metrics_accumulator.dataset_metric_names:
            gt_dir = os.path.dirname(gt_image_paths[0]) if gt_image_paths else None
            metrics_accumulator.compute_dataset_metrics(args.output_dir, gt_dir=gt_dir)
        metrics_accumulator.print_summary()
        metrics_output_path = os.path.join(args.output_dir, "metrics.json")
        metrics_accumulator.save(metrics_output_path)
        print(f"Metrics saved to {metrics_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for GeneratorEdit (Qwen-Image-Edit based)")
    parser.add_argument("--lq_input_txt", type=str, default=None,
                        help="Path to txt file containing LQ input image paths")
    parser.add_argument("--ref_input_txt", type=str, default=None,
                        help="Path to txt file containing reference image paths (optional)")
    parser.add_argument("--prompt_input_txt", type=str, default=None,
                        help="Path to txt file containing prompt file paths (optional)")
    parser.add_argument("--fidelity_mask_input_txt", type=str, default=None,
                        help="Path to txt file containing fidelity mask paths (grayscale PNG). "
                             "mask=0(black) uses --fidelity value, mask=255(white) uses fidelity=1.0 (preserve original), "
                             "intermediate values linearly interpolate.")
    parser.add_argument("--gt_input_txt", type=str, default=None,
                        help="Path to txt file containing GT image paths for evaluation")
    parser.add_argument(
        "--deg_file_path",
        type=str,
        default="./examples/qwen_image/configs/deg_pisa.yaml",
        help="The path of the deg yaml."
    )
    parser.add_argument("--trained_ckpt", type=str, required=True,
                        help="Path to trained GeneratorEdit checkpoint")
    parser.add_argument("--output_dir", type=str, default="./test_outputs_edit",
                        help="Path to save results")
    parser.add_argument("--scale", type=float, default=None,
                        help="SR scale factor (mutually exclusive with --target_pixels)")
    parser.add_argument("--target_pixels", type=int, default=None,
                        help="Target total pixel budget for output (e.g. 2073600 for ~1920x1080). "
                             "Output is proportionally scaled to meet w*h ≈ target_pixels. "
                             "Mutually exclusive with --scale")
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="Classifier-Free Guidance scale")
    parser.add_argument("--fidelity", type=float, default=1.0,
                        help="Fidelity scale")
    parser.add_argument("--mode", type=str, default="one_step", choices=["one_step", "multi_step"],
                        help="Inference mode: one_step (GAN style) or multi_step (standard diffusion)")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="Number of denoising steps in multi_step mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for multi_step mode")
    parser.add_argument("--start_end", type=str, default="0,",
                        help="Index range, e.g. '0,100'")
    parser.add_argument("--align_method", type=str, default="adain",
                        help="Color alignment method: adain, wavelet, or none")
    parser.add_argument("--metrics", type=str, default="",
                        help="Comma-separated metrics. Supported: psnr,ssim,lpips,niqe,musiq,clipiqa,maniqa")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="Crop border pixels before computing PSNR/SSIM")
    parser.add_argument("--gen_start_point", type=int, default=750,
                        help="Generator start timestep point for one_step mode")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA rank (must match the trained checkpoint)")
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
                        help="LoRA target modules (must match the trained checkpoint)")
    parser.add_argument("--ref_max_pixels", type=int, default=None,
                        help="Max total pixel budget for REF image. "
                             "Image is proportionally scaled so that w*h <= this value (optional)")
    parser.add_argument("--tiled", default=False, action="store_true",
                        help="Enable tiled inference for large images. "
                             "Each tile sees the full edit_latents (global context) but only denoises a local patch.")
    parser.add_argument("--tile_size", type=int, default=64,
                        help="Tile size in latent space (pixel space = tile_size * 8, default 64 = 512px)")
    parser.add_argument("--tile_stride", type=int, default=48,
                        help="Tile stride in latent space (< tile_size for overlap, default 48 = 384px)")
    parser.add_argument("--tile_prompt_mode", type=str, default="global", choices=["global", "tile"],
                        help="Prompt encoding mode for tiled inference: "
                             "'global' encodes prompt with the full image (faster), "
                             "'tile' encodes prompt per tile using cropped LQ region (matches training, slower)")
    parser.add_argument("--use_prompt_prefix", default=False, action="store_true",
                        help="Prepend edit prompt prefix to match training with use_edit_prompt_prefix=True")
    parser.add_argument("--input_min_short_edge", type=int, default=None,
                        help="If set, upscale LQ image so that its short edge >= this value before SR inference")
    parser.add_argument("--use_full_lq_condition", default=False, action="store_true",
                        help="Enable three-condition mode [lq_crop, full_lq, ref]. "
                             "In tiled mode, lq is auto-cropped per tile while full_lq stays global.")
    parser.add_argument("--full_lq_max_pixels", type=int, default=None,
                        help="Max total pixel budget for full_lq condition image (only effective with --use_full_lq_condition).")
    parser.add_argument("--drop_lq_crop_condition", default=False, action="store_true",
                        help="Drop LQ crop from edit_latents and prompt, only use full_lq + ref as conditions.")
    # SAM mask 相关参数
    parser.add_argument("--hq_ori_mask_txt_paths", type=str, default=None,
                        help="Path to txt file containing LQ mask paths (grayscale, white=need ref)")
    parser.add_argument("--ref_crop_mask_txt_paths", type=str, default=None,
                        help="Path to txt file containing REF mask paths (grayscale, white=valid texture)")
    parser.add_argument("--mask_low_noise_ratio", type=float, default=0.1,
                        help="Noise ratio for mask=0 regions (0.0=no noise, 1.0=same as mask=1)")
    # verify_mask 模式专用参数
    parser.add_argument("--verify_mask", default=False, action="store_true",
                        help="Run mask verification mode instead of normal test")
    parser.add_argument("--verify_gt", type=str, default=None,
                        help="GT image path for mask verification (will be downscaled 4x to get LQ)")
    parser.add_argument("--verify_prompt", type=str, default=None,
                        help="GT image prompt for mask verification")
    parser.add_argument("--verify_ref", type=str, default=None,
                        help="REF image path for mask verification")
    parser.add_argument("--verify_sam_lq_mask", type=str, default=None,
                        help="SAM LQ mask path for mask verification (grayscale, white=need ref)")
    parser.add_argument("--verify_sam_ref_mask", type=str, default=None,
                        help="SAM REF mask path for mask verification (grayscale, white=valid texture)")
    parser.add_argument("--verify_degrade", type=str, default="bicubic", choices=["bicubic", "realesrgan"],
                        help="Degradation method for verify_mask: bicubic (simple 4x down) or realesrgan (realistic)")
    args = parser.parse_args()

    if not args.verify_mask:
        # 正常 test 模式的校验
        if args.scale is not None and args.target_pixels is not None:
            parser.error("--scale and --target_pixels are mutually exclusive, please specify only one")
        if args.scale is None and args.target_pixels is None:
            args.scale = 1.0  # 兼容旧行为：都不指定时默认 1x
            print("Neither --scale nor --target_pixels specified, defaulting to --scale 1.0")

    return args


@torch.no_grad()
def verify_mask(args):
    """
    Mask 噪声调制验证：输入 GT 图，下采样 4x 得到 LQ，用 4 种 mask 配置对比输出。

    用法:
        python test_sr_edit.py --verify_mask \
            --trained_ckpt /path/to/ckpt \
            --verify_gt /path/to/test_gt.png \
            --verify_prompt /path/to/test_prompt.txt \
            --verify_ref /path/to/test_ref.png \
            --verify_sam_lq_mask /path/to/lq_mask.png \
            --verify_sam_ref_mask /path/to/ref_mask.png \
            --output_dir ./verify_mask_output \
            --verify_degrade bicubic
    """
    model = init_model(
        trained_ckpt=args.trained_ckpt,
        gen_start_point=args.gen_start_point,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
    )

    gt_pil = Image.open(args.verify_gt).convert("RGB")
    ref_pil = Image.open(args.verify_ref).convert("RGB")

    gt_w, gt_h = gt_pil.size

    # 下采样 4x 得到 LQ
    lq_w, lq_h = gt_w // 4, gt_h // 4
    degrade_method = getattr(args, 'verify_degrade', 'bicubic')
    verify_lq_path = args.verify_gt.replace('.png', f'_lq_{degrade_method}.png')
    if not os.path.exists(verify_lq_path):
        if degrade_method == 'bicubic':
            lq_pil = gt_pil.resize((lq_w, lq_h), Image.BICUBIC)
        else:
            # 使用 RealESRGAN 退化
            import numpy as np
            from diffsynth.extensions.realesrgan.realesrgan import RealESRGAN_degradation
            degradation = RealESRGAN_degradation(args.deg_file_path, device='cuda')
            gt_np = np.asarray(gt_pil).astype(np.float32) / 255.
            _, lq_t = degradation.degrade_process(gt_np, resize_bak=True)
            lq_t = lq_t.squeeze(0)
            lq_pil = Image.fromarray((lq_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8))
            lq_pil = lq_pil.resize((lq_w, lq_h), Image.BICUBIC)
        # 保存 LQ
        tmp_file = f"./tmp_lq_{degrade_method}.png"
        lq_pil.save(tmp_file)
        shutil.copy2(tmp_file, verify_lq_path)
        os.remove(tmp_file)
    else:
        lq_pil = Image.open(verify_lq_path).convert("RGB")

    print(f"GT: {gt_w}x{gt_h} -> LQ: {lq_w}x{lq_h} (4x downscale, method={degrade_method})")

    # 读取 prompt
    prompt = "High quality, hyper detailed photo"
    if args.verify_prompt:
        if os.path.exists(args.verify_prompt):
            with open(args.verify_prompt, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if content:
                prompt = content
    if args.use_prompt_prefix:
        prompt = (
            "Enhance the low-quality image to high resolution with sharp details and fine textures, "
            "using the high-quality reference image as a guide for texture and appearance: "
        ) + prompt

    os.makedirs(args.output_dir, exist_ok=True)

    

    common_kwargs = dict(
        model=model,
        lq_pil=lq_pil,
        prompt=prompt,
        ref_pil=ref_pil,
        scale=4.0,
        target_pixels=None,
        cfg_scale=args.cfg,
        tiled=args.tiled,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        use_prompt_prefix=False,
        use_full_lq_condition=args.use_full_lq_condition,
        full_lq_max_pixels=args.full_lq_max_pixels,
        ref_max_pixels=args.ref_max_pixels,
        align_method=args.align_method,
        mask_low_noise_ratio=args.mask_low_noise_ratio,
    )

    # A: 全白 mask（全部参考 ref）
    # 注意：inject_mask_latent=False 避免 5 张 edit_image 导致 prompt encode 卡死
    # 仅验证噪声调制效果；mask latent 注入需训练后单独验证
    print("[A] All-white mask (full ref guidance, noise modulation only)...")
    white_lq_mask = Image.new('L', lq_pil.size, 255)
    white_ref_mask = Image.new('L', ref_pil.size, 255)
    result_a = infer_single(**common_kwargs, lq_mask_pil=white_lq_mask, ref_mask_pil=white_ref_mask,
                            inject_mask_latent=False)
    result_a.save(os.path.join(args.output_dir, "A_all_white_mask.png"))
    print("  Saved A_all_white_mask.png")

    # B: 全黑 mask（全部不参考 ref）
    print("[B] All-black mask (no ref guidance, noise modulation only)...")
    black_lq_mask = Image.new('L', lq_pil.size, 0)
    black_ref_mask = Image.new('L', ref_pil.size, 0)
    result_b = infer_single(**common_kwargs, lq_mask_pil=black_lq_mask, ref_mask_pil=black_ref_mask,
                            inject_mask_latent=False)
    result_b.save(os.path.join(args.output_dir, "B_all_black_mask.png"))
    print("  Saved B_all_black_mask.png")

    # C: 真实 SAM mask
    print("[C] Real SAM mask (noise modulation only)...")
    sam_lq_mask = Image.open(args.verify_sam_lq_mask).convert("L")
    sam_ref_mask = Image.open(args.verify_sam_ref_mask).convert("L")
    result_c = infer_single(**common_kwargs, lq_mask_pil=sam_lq_mask, ref_mask_pil=sam_ref_mask,
                            inject_mask_latent=False)
    result_c.save(os.path.join(args.output_dir, "C_sam_mask.png"))
    print("  Saved C_sam_mask.png")

    # D: 无 mask（原始行为基线）
    print("[D] No mask (baseline)...")
    result_d = infer_single(**common_kwargs, lq_mask_pil=None, ref_mask_pil=None)
    result_d.save(os.path.join(args.output_dir, "D_baseline_no_mask.png"))
    print("  Saved D_baseline_no_mask.png")

    # E: 真实 SAM mask + 仅噪声调制（不注入 mask latent，保持原始 edit_latents 结构）
    print("[E] SAM mask noise modulation only (no mask latent injection)...")
    result_e = infer_single(**common_kwargs, lq_mask_pil=sam_lq_mask, ref_mask_pil=sam_ref_mask,
                            inject_mask_latent=False)
    result_e.save(os.path.join(args.output_dir, "E_noise_modulation_only.png"))
    print("  Saved E_noise_modulation_only.png")

    # 拼接对比图
    print("Generating comparison grid...")
    results = [result_a, result_b, result_c, result_d, result_e]

    # 统一到 GT 尺寸
    target_w, target_h = gt_w, gt_h
    results = [r.crop((0, 0, target_w, target_h)) if r.size[0] >= target_w and r.size[1] >= target_h
               else r.resize((target_w, target_h), Image.BICUBIC) for r in results]

    # 第 1 行：GT | LQ(上采样) | REF | LQ mask | REF mask | (空白占位)
    upsampled_lq = lq_pil.resize((target_w, target_h), Image.BICUBIC)
    gt_display = gt_pil.resize((target_w, target_h), Image.BICUBIC)
    ref_display = ref_pil.resize((target_w, target_h), Image.BICUBIC)
    lq_mask_display = sam_lq_mask.convert('RGB').resize((target_w, target_h), Image.BILINEAR)
    ref_mask_display = sam_ref_mask.convert('RGB').resize((target_w, target_h), Image.BILINEAR)
    blank = Image.new('RGB', (target_w, target_h), (128, 128, 128))
    row1 = [gt_display, upsampled_lq, ref_display, lq_mask_display, ref_mask_display, blank]

    # 第 2 行：A(全白) | B(全黑) | C(SAM) | D(基线) | E(仅噪声调制) | GT(对照)
    row2 = results + [gt_display]

    num_cols = 6
    total_canvas_w = target_w * num_cols
    total_canvas_h = target_h * 2
    canvas = Image.new('RGB', (total_canvas_w, total_canvas_h))
    for i, img in enumerate(row1):
        canvas.paste(img, (i * target_w, 0))
    for i, img in enumerate(row2):
        canvas.paste(img, (i * target_w, target_h))

    canvas.save(os.path.join(args.output_dir, "comparison_grid.png"))
    print(f"\nComparison grid saved to {os.path.join(args.output_dir, 'comparison_grid.png')}")
    print(f"Row 1: GT | LQ(bicubic up) | REF | LQ mask | REF mask | (blank)")
    print(f"Row 2: A(white) | B(black) | C(SAM) | D(baseline) | E(noise-mod only) | GT")
    print(f"\nVerification criteria:")
    print(f"  - D     : baseline (no mask)")
    print(f"  - E vs D: noise modulation effect without latent injection")
    print(f"  - C vs E: effect of mask latent injection (needs training to work)")


if __name__ == '__main__':
    args = parse_args()
    if args.verify_mask:
        verify_mask(args)
    else:
        test(args)
