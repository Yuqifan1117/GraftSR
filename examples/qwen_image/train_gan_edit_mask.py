"""
基于原生 Qwen-Image-Edit 的 GAN 训练脚本。

与原版 train_gan.py 的核心区别：
- Generator 使用 GeneratorEdit（原生 model_fn + 标准 LoRA + zero_cond_t）
- 条件图通过 edit_latents 传入，由 zero_cond_t 机制自动处理
- 仅对条件流中的 LQ 块加噪（与 base 模型一致），full_lq 和 ref 不加噪
- 移除所有 DualLoRA/ConditionTypeEmbedding 相关逻辑
- 保留完整的 GAN 训练策略（Discriminator、GAN loss、LPIPS 等）
- GAN loss 通过 gan_w_ratio 根据 LQ 条件噪声水平动态加权

单步 gan训练
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from accelerate import Accelerator

# 1. 尽早初始化 Accelerator
accelerator = Accelerator()

# 2. 只让每台机器的 0 号显卡进程去修改文件和编译代码
if accelerator.is_local_main_process:
    print(f"[{os.uname().nodename}] 主进程开始修复 basicsr 环境并编译项目...")
    
    # 指令 1：修复系统级 Conda 环境下的 basicsr (加了 2>/dev/null 屏蔽找不到文件的报错)
    os.system("sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /opt/conda/envs/python3.10.13/lib/python3.10/site-packages/basicsr/data/degradations.py 2>/dev/null")
    
    # 指令 2：修复用户级目录下的 basicsr
    os.system("sed -i '8s/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' /root/.local/lib/python3.10/site-packages/basicsr/data/degradations.py 2>/dev/null")
    
    # 指令 3：执行 setup.py develop
    # 【重点优化】：将 'python' 替换为 sys.executable，确保使用的是当前正确的 Python 环境
    # 【注意路径】：如果 setup.py 不在当前执行命令的目录下，需替换为正确的绝对路径或相对路径
    os.system(f"{sys.executable} setup.py develop")
    
    print(f"[{os.uname().nodename}] 环境修复与编译完成！")

# 3. 拦截所有其他进程，必须等主进程把上面的 `sed` 和 `setup.py` 跑完！
accelerator.wait_for_everyone()
import math
import random
import shutil
import torch
import torchvision
import numpy as np
torch.backends.cuda.enable_cudnn_sdp(False)
from datetime import datetime
from contextlib import redirect_stdout, nullcontext

from utils import yaml_load, parse_args, _save_image
from PIL import Image

from ganloss import GANLoss
from generator_edit import GeneratorEdit
from discriminator import Discriminator
from check_lpips_cache import ensure_dists_cache, ensure_musiq_cache, ensure_vgg_cache
from test_sr import read_image_paths_from_txt
from metrics import MetricsAccumulator, distributed_gather_metrics

from diffsynth.extensions.realesrgan.dataset import PairedSROnlineTxtDataset
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from lightning.fabric.loggers import TensorBoardLogger

import lpips
import pyiqa
import matplotlib.pyplot as plt


def _safe_unwrap(obj):
    """Safe accessor: handles both DDP-wrapped and unwrapped modules."""
    return obj.module if hasattr(obj, 'module') else obj


def get_gan_weight(now_sigma, start_sigma, min_weight=1.0, max_weight=5.0):
    """根据条件流 LQ 噪声水平动态调整 GAN loss 权重"""
    if start_sigma == 0:
        return min_weight
    ratio = now_sigma / start_sigma
    weight = min_weight + (max_weight - min_weight) * ratio
    return weight


def deal_discriminator_condition(input1, input2, use_dual_condition_flag):
    if use_dual_condition_flag:
        return torch.stack([input1, input2], dim=2)
    else:
        return input1


def adaptive_relaxed_mean(tensor, output_size=10):
    if output_size >= 100:
        return tensor
    else:
        orig_size = tensor.shape[2:]
        pooled = torch.nn.functional.adaptive_avg_pool2d(tensor, (output_size, output_size))
        return torch.nn.functional.interpolate(pooled, size=orig_size, mode='bilinear')


def build_pixel_loss_fn(loss_type='mse'):
    """根据配置构建像素级损失函数"""
    if loss_type == 'l1':
        return torch.nn.functional.l1_loss
    elif loss_type == 'charbonnier':
        def charbonnier_loss(pred, target, eps=1e-6):
            return torch.mean(torch.sqrt((pred - target) ** 2 + eps ** 2))
        return charbonnier_loss
    else:
        return torch.nn.functional.mse_loss


def add_gaussian_noise(input_tensor, variance=0.01):
    std = variance ** 0.5
    noise = torch.randn_like(input_tensor) * std
    return input_tensor + noise


def random_crop_pair(data_dict, crop_h, crop_w):
    """对包含 'gt' 和 'lq' (以及可选 'lq_for_cond') 的字典进行相同的随机裁剪。
    
    如果 data_dict 中包含 'lq_for_cond'，会对其做同样裁剪并写回 data_dict。
    """
    gt_img = data_dict['gt']
    lq_img = data_dict['lq']

    assert gt_img.size == lq_img.size, "GT 和 LQ 的尺寸不一致"
    w, h = gt_img.size
    assert crop_w <= w and crop_h <= h, "裁剪尺寸不能大于原图尺寸"

    left = random.randint(0, w - crop_w)
    top = random.randint(0, h - crop_h)
    box = (left, top, left + crop_w, top + crop_h)

    # 如果有 lq_for_cond，也做同样裁剪
    if 'lq_for_cond' in data_dict:
        lq_for_cond_img = data_dict['lq_for_cond']
        assert lq_for_cond_img.size == gt_img.size, "lq_for_cond 与 GT 尺寸不一致"
        data_dict['lq_for_cond'] = lq_for_cond_img.crop(box)

    return gt_img.crop(box), lq_img.crop(box)


@torch.no_grad()
def distributed_test(accelerator, generator, epoch, workdir,
                     test_lq_txt, test_ref_txt=None, test_prompt_txt=None, test_gt_txt=None,
                     test_scale=2.0, test_cfg=1.0,
                     test_metrics='', test_crop_border=0,
                     use_full_lq_condition=False,
                     ref_max_pixels=None,
                     full_lq_max_pixels=None,
                     drop_lq_crop_condition=False):
    """每个 epoch 结束后的分布式测试（适配 GeneratorEdit 的 infer_one_step）"""

    image_paths = read_image_paths_from_txt(test_lq_txt)
    ref_image_paths = read_image_paths_from_txt(test_ref_txt) if test_ref_txt else []
    prompt_paths = read_image_paths_from_txt(test_prompt_txt) if test_prompt_txt else []
    gt_image_paths = read_image_paths_from_txt(test_gt_txt) if test_gt_txt else []

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    local_indices = list(range(rank, len(image_paths), world_size))

    gen_module = accelerator.unwrap_model(generator)
    gen_module.eval()
    gen_module.device = accelerator.device
    gen_module.pipe.device = accelerator.device

    test_output_dir = os.path.join(workdir, "test", f"epoch_{epoch}")
    os.makedirs(test_output_dir, exist_ok=True)
    viz_dir = os.path.join(test_output_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    metrics_accumulator = None
    metric_names = [m.strip() for m in test_metrics.split(",") if m.strip()] if test_metrics else []
    if metric_names:
        metrics_accumulator = MetricsAccumulator(
            metrics=metric_names,
            device=str(accelerator.device),
            crop_border=test_crop_border,
        )
        accelerator.print(f"Evaluation metrics enabled: {metric_names}, crop_border={test_crop_border}")

    for local_idx, global_idx in enumerate(local_indices):
        img_path = image_paths[global_idx]
        filename_without_extension = os.path.splitext(os.path.basename(img_path))[0]

        # 加载 LQ
        lq_pil = Image.open(img_path).convert("RGB")
        target_w = int(lq_pil.width * test_scale)
        target_h = int(lq_pil.height * test_scale)
        # 上采样 LQ 到目标尺寸
        upsampled_lq = lq_pil.resize((target_w, target_h), Image.BICUBIC)

        # 加载 REF（如果有）
        ref_pil = None
        if ref_image_paths:
            ref_pil = Image.open(ref_image_paths[global_idx]).convert("RGB")
            if ref_max_pixels is not None:
                from test_sr_edit import apply_pixel_budget
                ref_pil = apply_pixel_budget(ref_pil, max_pixels=ref_max_pixels)

        # 加载 prompt（如果有）
        prompt = ""
        if prompt_paths:
            prompt_path = prompt_paths[global_idx]
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()

        # 准备 full_lq 条件图（use_full_lq_condition 控制是否启用，full_lq_max_pixels 控制分辨率预算）
        full_lq_pil = None
        if use_full_lq_condition:
            from test_sr_edit import apply_pixel_budget
            full_lq_pil = apply_pixel_budget(upsampled_lq, max_pixels=full_lq_max_pixels)

        # 推理：使用 infer_one_step
        res_img = gen_module.infer_one_step(
            prompt=prompt,
            negative_prompt="",
            lq_pil=upsampled_lq,
            ref_pil=ref_pil,
            full_lq_pil=full_lq_pil,
            cfg_scale=test_cfg,
            drop_lq_crop_condition=drop_lq_crop_condition,
        )

        # 裁剪到目标尺寸并保存
        res_img = res_img.resize((target_w, target_h), Image.BICUBIC)
        output_path = os.path.join(test_output_dir, f"{filename_without_extension}.png")
        res_img.save(output_path)

        # 可视化：拼接 LQ | Output | GT(如有) | REF(如有)
        viz_imgs = [upsampled_lq, res_img]
        if gt_image_paths:
            gt_pil = Image.open(gt_image_paths[global_idx]).convert("RGB")
            gt_pil = gt_pil.resize((target_w, target_h), Image.BICUBIC)
            viz_imgs.insert(0, gt_pil)
        if ref_pil is not None:
            viz_imgs.append(ref_pil)

        target_viz_h = viz_imgs[0].height
        resized_viz = []
        for im in viz_imgs:
            if im.height != target_viz_h:
                scale_factor = target_viz_h / im.height
                im = im.resize((int(im.width * scale_factor), target_viz_h), Image.BICUBIC)
            resized_viz.append(im)
        total_viz_w = sum(im.width for im in resized_viz)
        canvas = Image.new('RGB', (total_viz_w, target_viz_h))
        x_offset = 0
        for im in resized_viz:
            canvas.paste(im, (x_offset, 0))
            x_offset += im.width
        canvas.save(os.path.join(viz_dir, f"{filename_without_extension}.png"))

        # 评测指标
        if metrics_accumulator is not None:
            gt_img_for_metric = None
            if gt_image_paths:
                gt_img_for_metric = Image.open(gt_image_paths[global_idx]).convert("RGB")
                gt_img_for_metric = gt_img_for_metric.resize((target_w, target_h), Image.BICUBIC)
            per_image_metrics = metrics_accumulator.update(res_img, gt_img_for_metric, image_name=filename_without_extension)
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in per_image_metrics.items() if k != "image_name")
            accelerator.print(f"[Rank {rank}] Metrics: {metrics_str}")

        accelerator.print(f"[Rank {rank}] Test [{local_idx+1}/{len(local_indices)}] {filename_without_extension}")

    if metrics_accumulator is not None:
        gt_dir = os.path.dirname(gt_image_paths[0]) if gt_image_paths else None
        distributed_gather_metrics(metrics_accumulator, accelerator, test_output_dir, gt_dir=gt_dir)
    else:
        accelerator.wait_for_everyone()

    accelerator.print(f"Epoch {epoch} distributed test done. Results saved to {test_output_dir}")
    gen_module.train()

def downsample_mask(data, mask_name, accelerator):
    mask_pil = data.get(mask_name, None)
    if mask_pil is not None:
        w, h = mask_pil.size
        resized_mask = mask_pil.resize((w//8, h//8), Image.Resampling.BOX)
        arr = np.array(resized_mask)
        arr = (arr > 0).astype(np.uint8) * 255
        resized_mask = Image.fromarray(arr, 'L')
        mask_tensor = (torchvision.transforms.ToTensor()(resized_mask)>0).to(accelerator.device)
        return mask_pil, mask_tensor
    else:
        return None, None

def apply_gt_postprocess(gt_pil, contrast=1.02, saturation=1.03, sharpness=1.05, jpeg_quality=92):
    """
    对 GT PIL Image 应用自然化后处理（数据蒸馏）。
    使 DiT 直接学习自然化后的图像分布，消除生成图与自然照片的统计差异。

    Args:
        gt_pil: PIL Image (RGB)
        contrast: 对比度因子
        saturation: 饱和度因子
        sharpness: 锐度因子
        jpeg_quality: JPEG 压缩质量（None 或 0 表示不做）
    Returns:
        处理后的 PIL Image
    """
    from PIL import ImageEnhance
    import io

    if contrast != 1.0:
        gt_pil = ImageEnhance.Contrast(gt_pil).enhance(contrast)
    if saturation != 1.0:
        gt_pil = ImageEnhance.Color(gt_pil).enhance(saturation)
    if sharpness != 1.0:
        gt_pil = ImageEnhance.Sharpness(gt_pil).enhance(sharpness)
    if jpeg_quality and jpeg_quality > 0:
        buffer = io.BytesIO()
        gt_pil.save(buffer, format="JPEG", quality=jpeg_quality)
        buffer.seek(0)
        gt_pil = Image.open(buffer).convert("RGB")
    return gt_pil


def encode_img_from_data(data, name, generator, accelerator):
    img_pil = data.get(name, None)
    if img_pil is not None:
        img_rgb = _safe_unwrap(generator).pipe.preprocess_image(img_pil).to(
            device=accelerator.device, dtype=torch.bfloat16
        )
        img_latents = _safe_unwrap(generator).pipe.vae.encode(img_rgb, tiled=False).detach()
        return img_pil, img_rgb, img_latents
    else:
        return None, None, None


def train(args):
    ensure_vgg_cache()
    ensure_dists_cache()
    ensure_musiq_cache()

    dataset_yaml = yaml_load(args.mmaigc_dataset_yml)
    gradient_accumulation_steps = dataset_yaml['accumulate_grad_batches']
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    enable_tensorboard = args.enable_tensorboard
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='bf16',
        log_with='tensorboard' if enable_tensorboard else None,
        project_dir=os.path.join(args.output_dir, dataset_yaml['exp_tag']) if enable_tensorboard else None,
        kwargs_handlers=[ddp_kwargs],
    )

    set_seed(42)

    dataset = PairedSROnlineTxtDataset(split="train", args=args)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=1, num_workers=2,
        collate_fn=lambda x: x[0]
    )

    pretrained_qwen_path = os.environ["qwen_path"]
    pretrained_wan_path = os.environ["wan_path"]

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

    gen_start_point = dataset_yaml.get('gen_start_point', 750)

    # ======== 使用 GeneratorEdit 替代 Generator ========
    generator = GeneratorEdit(
        torch_dtype=torch.bfloat16,
        pretrained_weights=sd_safe_tensor_path_json_format,
        tokenizer_path=f"{pretrained_qwen_path}/tokenizer",
        processor_path=f"{pretrained_qwen_path}/processor" if os.path.exists(f"{pretrained_qwen_path}/processor") else None,
        learning_rate=dataset_yaml['learning_rate'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_gen=dataset_yaml['pretrained_ckpt_path_gen'],
        gen_start_point=gen_start_point,
        train_new_vae=dataset_yaml.get('new_vae_w', 1.0) > 0,
        lora_rank=dataset_yaml.get('lora_rank', 128),
        lora_target_modules=dataset_yaml.get(
            'lora_target_modules',
            "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"
        ),
        zero_cond_t=dataset_yaml.get('zero_cond_t', True),
    )

    # ======== Discriminator 保持不变 ========
    if args.offload_dis_t5:
        dis_pretrained_weights = f"""[
            "{pretrained_wan_path}/diffusion_pytorch_model.safetensors",
            "{pretrained_wan_path}/Wan2.1_VAE.pth"
        ]"""
    else:
        dis_pretrained_weights = f"""[
            "{pretrained_wan_path}/diffusion_pytorch_model.safetensors",
            "{pretrained_wan_path}/models_t5_umt5-xxl-enc-bf16.pth",
            "{pretrained_wan_path}/Wan2.1_VAE.pth"
        ]"""

    discriminator = Discriminator(
        torch_dtype=torch.bfloat16,
        pretrained_weights=dis_pretrained_weights,
        dis_tokenizer_path=f'{pretrained_wan_path}/google/umt5-xxl',
        learning_rate=dataset_yaml['learning_rate_dis'],
        use_gradient_checkpointing=dataset_yaml['use_gradient_checkpointing'],
        pretrained_ckpt_path_dis=dataset_yaml['pretrained_ckpt_path_dis']
    )

    # ======== Loss / Optimizer / Scheduler ========
    cri_gan = GANLoss(
        gan_type=dataset_yaml['gan_type'],
        loss_weight=dataset_yaml['gan_loss_weight'],
        real_label_val=dataset_yaml['real_label_val'],
        fake_label_val=dataset_yaml['fake_label_val'],
    )

    optimizer_g = generator.configure_optimizers()
    optimizer_d = discriminator.configure_optimizers()

    lr_scheduler_type = dataset_yaml.get('lr_scheduler_type', 'none')
    lr_eta_min_g = dataset_yaml.get('lr_scheduler_eta_min_g', 1e-6)
    lr_eta_min_d = dataset_yaml.get('lr_scheduler_eta_min_d', 1e-7)
    total_training_steps = dataset_yaml['max_epochs'] * math.ceil(len(dataloader) / gradient_accumulation_steps)
    scheduler_g, scheduler_d = None, None
    if lr_scheduler_type == 'cosine':
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_g, T_max=total_training_steps, eta_min=lr_eta_min_g
        )
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_d, T_max=total_training_steps, eta_min=lr_eta_min_d
        )
    elif lr_scheduler_type == 'linear':
        scheduler_g = torch.optim.lr_scheduler.LinearLR(
            optimizer_g, start_factor=1.0, end_factor=lr_eta_min_g / dataset_yaml['learning_rate'],
            total_iters=total_training_steps
        )
        scheduler_d = torch.optim.lr_scheduler.LinearLR(
            optimizer_d, start_factor=1.0, end_factor=lr_eta_min_d / dataset_yaml['learning_rate_dis'],
            total_iters=total_training_steps
        )

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    # 新增 DISTS 感知损失
    dists_w = dataset_yaml.get('dists_w', 0.0)
    net_dists = None
    if dists_w > 0:
        net_dists = pyiqa.create_metric('dists', device=accelerator.device)
        net_dists.requires_grad_(False)
        net_dists.eval()


    generator.pipe.device = accelerator.device
    discriminator.pipe.device = accelerator.device

    if net_dists is not None:
        generator, discriminator, net_lpips, net_dists, optimizer_g, optimizer_d, dataloader = accelerator.prepare(
            generator, discriminator, net_lpips, net_dists, optimizer_g, optimizer_d, dataloader
        )
    else:
        generator, discriminator, net_lpips, optimizer_g, optimizer_d, dataloader = accelerator.prepare(
            generator, discriminator, net_lpips, optimizer_g, optimizer_d, dataloader
        )

    # ---- 主进程创建 workdir 并初始化日志 ----
    if accelerator.is_main_process:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        workdir = os.path.join(args.output_dir, dataset_yaml['exp_tag'], current_time)
        print(f"workdir: {workdir}")
        os.makedirs(workdir, exist_ok=False)
        trt_logger = TensorBoardLogger(workdir, name="tensorboard") if enable_tensorboard else None

        # backup config file
        yaml_file_name = os.path.basename(args.mmaigc_dataset_yml)
        target_path = os.path.join(workdir, yaml_file_name)
        shutil.copy2(args.mmaigc_dataset_yml, target_path)

        # save args to file
        args_file_name = os.path.join(workdir, 'args.txt')
        with open(args_file_name, "w", encoding='utf-8') as f:
            for arg, value in vars(args).items():
                f.write(f"{arg}: {value}\n")

        # plot model structure
        filename = os.path.join(workdir, 'generator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(_safe_unwrap(generator).pipe)

        filename = os.path.join(workdir, 'discriminator_model_structure.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print(_safe_unwrap(discriminator).pipe)

        # 获取生成器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'generator_trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                for name, param in generator.named_parameters():
                    if param.requires_grad:
                        print(name)

        # 获取判别器可训练参数名字并写入文件
        filename = os.path.join(workdir, 'discriminator_trainable_parameters.txt')
        with open(filename, "a") as f:
            with redirect_stdout(f):
                print("Trainable parameters in discriminator:")
                for name, param in discriminator.named_parameters():
                    if param.requires_grad:
                        print(name)

        # plot scheduler sigmas
        data_np = _safe_unwrap(generator).pipe.scheduler.sigmas.cpu().numpy()
        plt.plot(data_np, marker='o')
        plt.title('Line Plot of sigmas')
        plt.xlabel('timesteps_id')
        plt.ylabel('sigma')
        plt.grid(True)
        plt.savefig(os.path.join(workdir, 'line_plot_G.png'))

        plt.clf()

        data_np = _safe_unwrap(discriminator).pipe.scheduler.sigmas.cpu().numpy()
        plt.plot(data_np, marker='o')
        plt.title('Line Plot of sigmas')
        plt.xlabel('timesteps_id')
        plt.ylabel('sigma')
        plt.grid(True)
        plt.savefig(os.path.join(workdir, 'line_plot_D.png'))
    else:
        workdir = ""

    # 广播 workdir 到所有进程
    workdir_list = [workdir]
    torch.distributed.broadcast_object_list(workdir_list, src=0)
    workdir = workdir_list[0]

    rgb_w = dataset_yaml['rgb_w']
    lpips_w = dataset_yaml['lpips_w']
    new_vae_w = dataset_yaml.get('new_vae_w', 1.0)
    pixel_loss_fn = build_pixel_loss_fn(dataset_yaml.get('pixel_loss_type', 'mse'))

    iteration = 0
    log_iteration = 0

    # 记录判别器需要训练的参数的名字
    dis_trainable_param_names = [
        name for name, param in discriminator.named_parameters()
        if param.requires_grad
    ]

    for epoch in range(dataset_yaml['max_epochs']):
        total_batches = len(dataloader)
        for batch_idx, data in enumerate(dataloader, 0):
            total_iterations = dataset_yaml['max_epochs'] * total_batches
            current_progress = epoch * total_batches + batch_idx
            progress_percent = (current_progress / total_iterations) * 100
            epoch_progress = (batch_idx / total_batches) * 100

            accelerator.print(
                f"[Epoch {epoch}/{dataset_yaml['max_epochs']}] "
                f"[Batch {batch_idx}/{total_batches}] "
                f"[Iter {iteration}] "
                f"Progress: {epoch_progress:.1f}% (epoch) | {progress_percent:.1f}% (total)"
            )

            # =====================================================================
            # 预处理：编码 GT/LQ/REF，对条件流 LQ 块加噪，组装 edit_latents
            # =====================================================================
            with torch.no_grad():
                assert 'gt' in data and 'lq' in data and data['gt'].size == (512, 512) and data['lq'].size == (512, 512)

                gt_pil, gt_rgb, gt_latents = encode_img_from_data(data, 'gt', generator, accelerator)
                lq_pil, lq_rgb, lq_latents = encode_img_from_data(data, 'lq', generator, accelerator)
                full_lq_pil, full_lq_rgb, full_lq_latents = encode_img_from_data(data, 'full_lq', generator, accelerator) # 当前实验不需要，编码完整 LQ（三条件模式，如果有）
                ref_pil, ref_rgb, ref_latents = encode_img_from_data(data, 'ref', generator, accelerator) # 编码 REF（如果有）


                # Prior Noise 用于增强图
                gen_start_point = _safe_unwrap(generator).gen_start_point
                fixed_timestep_id = torch.randint(gen_start_point, gen_start_point + 1, (1,))
                fixed_timestep = _safe_unwrap(generator).pipe.scheduler.timesteps[fixed_timestep_id].to(device=accelerator.device)
                one_step_sigma = _safe_unwrap(generator).pipe.scheduler.sigmas[fixed_timestep_id].to(
                    dtype=torch.bfloat16, device=accelerator.device
                )

                # ======== 仅对条件流 LQ 块加噪，full_lq 和 ref 不加噪 ========
                # Control Noise
                lq_cond_start_point = dataset_yaml.get('lq_cond_start_point', gen_start_point)
                random_timestep_id_for_lq_stream = torch.randint(lq_cond_start_point, 1000, (1,))
                random_timestep_for_lq_stream = _safe_unwrap(generator).pipe.scheduler.timesteps[
                    random_timestep_id_for_lq_stream
                ].to(device=accelerator.device)
                one_step_sigma_lq_stream = _safe_unwrap(generator).pipe.scheduler.sigmas[
                    random_timestep_id_for_lq_stream
                ].to(dtype=torch.bfloat16, device=accelerator.device)

                
                
                lq_noise = torch.randn_like(lq_latents)
                noisy_lq_latents = _safe_unwrap(generator).pipe.scheduler.add_noise(
                    lq_latents.detach(), lq_noise, random_timestep_for_lq_stream
                )
                gan_w_ratio = get_gan_weight(one_step_sigma_lq_stream, one_step_sigma)

                # ======== SAM Mask 噪声调制 + Mask 条件注入 ========
                # ---- Mask Dropout: 一定概率丢弃 mask，退化为原始训练 ----
                drop_mask = (random.random() < dataset_yaml.get('mask_dropout_prob', 0.0))
                if 'ref_mask' in data:
                    # 读取并下采样 LQ_MASK, REF_MASK（如果有）
                    lq_mask_pil, lq_mask = downsample_mask(data, 'lq_mask', accelerator)
                    ref_mask_pil, ref_mask = downsample_mask(data, 'ref_mask', accelerator)  

                    # latent 2D mask 用于条件 LQ 噪声调制 [B, 1, H_lat, W_lat]
                    lat_h, lat_w = lq_latents.shape[2], lq_latents.shape[3]
                    noise_mask_2d_resized = data["lq_mask"].resize((lat_w, lat_h), Image.BILINEAR)
                    noise_mask_2d = torchvision.transforms.functional.to_tensor(noise_mask_2d_resized).unsqueeze(0)
                    noise_mask_2d = noise_mask_2d.to(device=accelerator.device, dtype=torch.bfloat16)

                    if drop_mask:
                        lq_mask = None
                        ref_mask = torch.ones_like(ref_mask)
                        noise_mask_2d = None
                else:
                    lq_mask_pil, lq_mask = None, None
                    ref_mask_pil, ref_mask = None, None
                    noise_mask_2d = None


                # 一定概率使用纯 LQ 不加噪
                if random.random() < dataset_yaml.get('lq_no_noise_prob', 0.0):
                    noisy_lq_latents = lq_latents
                    gan_w_ratio = get_gan_weight(0, one_step_sigma)

                # 一定概率完全丢弃 LQ
                drop_lq = (data.get("dropout_lq", False)
                           and random.random() < dataset_yaml.get('lq_dropout_prob', 0.0))
                if drop_lq:
                    lq_mask = None
                    full_lq_latents = None
                    gan_w_ratio = get_gan_weight(one_step_sigma, one_step_sigma)

                # 组装 edit_latents：
                # 顺序: [lq(加噪) | lq_mask | full_lq | ref ｜ ref_mask ]
                edit_latents_parts = [] 
                edit_masks_parts = []  # 新建一个影子列表

                # 1. LQ 逻辑
                if not drop_lq:
                    edit_latents_parts.append(noisy_lq_latents)
                    edit_masks_parts.append(None)  # LQ 不需要 mask，塞一个 None 占位

                # 2. Full LQ 逻辑
                if full_lq_latents is not None:
                    edit_latents_parts.append(full_lq_latents)
                    edit_masks_parts.append(None)  # Full LQ 也不需要 mask，占位

                # 3. Ref 逻辑
                if ref_latents is not None:
                    edit_latents_parts.append(ref_latents)
                    if ref_mask is not None:
                        edit_masks_parts.append(ref_mask)
                    else:
                        edit_masks_parts.append(None)

                if len(edit_latents_parts) == 0:
                    edit_latents = None
                elif len(edit_latents_parts) == 1:
                    edit_latents = edit_latents_parts[0]
                else:
                    edit_latents = edit_latents_parts

                # 编码 prompt（edit_image 数量需与 edit_latents 一致）
                is_condition_dropped = data.get("dropout_ref", False)
                full_text = data["text"]
                raw_text = data.get("raw_text", "")
                if is_condition_dropped or random.random() < 0.75:
                    prompt_text = full_text
                else:
                    if raw_text and full_text.endswith(raw_text):
                        prompt_text = full_text[:-len(raw_text)].rstrip()
                    else:
                        prompt_text = full_text

                prompt_edit_images = [] if drop_lq else [lq_pil]
                if full_lq_pil is not None:
                    prompt_edit_images.append(full_lq_pil)
                if ref_pil is not None:
                    prompt_edit_images.append(ref_pil)
                if lq_mask_pil is not None:
                    prompt_edit_images.append(lq_mask_pil.convert("RGB"))
                edit_image_for_prompt = prompt_edit_images if len(prompt_edit_images) > 1 else prompt_edit_images[0]
                prompt_result = _safe_unwrap(generator).encode_prompt(prompt_text, edit_image=edit_image_for_prompt)
                pre_saved_prompt_emb = {
                    'prompt_emb': prompt_result['prompt_emb'],
                    'prompt_emb_mask': prompt_result['prompt_emb_mask'],
                }

                # 判别器 prompt
                caption = data.get('raw_text', data['text'])
                if args.offload_dis_t5:
                    raise NotImplementedError("not support offload_dis_t5")
                else:
                    dis_pipe = _safe_unwrap(discriminator).pipe
                    ids, mask = dis_pipe.tokenizer(caption, return_mask=True, add_special_tokens=True)
                    ids = ids.to(accelerator.device)
                    mask = mask.to(accelerator.device)
                    seq_lens = mask.gt(0).sum(dim=1).long()
                    prompt_emb = dis_pipe.text_encoder(ids, mask)
                    for i, v in enumerate(seq_lens):
                        prompt_emb[:, v:] = 0

            # =====================================================================
            # Train G
            # =====================================================================
            for name, param in discriminator.named_parameters():
                if name in dis_trainable_param_names:
                    param.requires_grad = False

            sync_gradients = ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx == len(dataloader) - 1)

            def make_ctx_g():
                return accelerator.no_sync(generator) if not sync_gradients else nullcontext()

            # ---- new_vae 前向传播（始终执行，因为主链路需要）----
            new_lq_latents = _safe_unwrap(generator).pipe.new_vae.encode(lq_rgb, tiled=False)
            new_lq_latents_rgb = _safe_unwrap(generator).pipe.vae.decode(new_lq_latents)
            
            # ---- new_vae loss: 仅当权重 > 0 时计算并反向传播 ----
            if new_vae_w > 0:
                with make_ctx_g():
                    loss_new_vae_lq = new_vae_w * pixel_loss_fn(new_lq_latents_rgb.float(), gt_rgb.float())
                    loss_new_vae_lq_scaled = loss_new_vae_lq / gradient_accumulation_steps
                    accelerator.backward(loss_new_vae_lq_scaled)
            else:
                loss_new_vae_lq = torch.tensor(0.0, device=accelerator.device)

            # ---- 主链路: new_lq_latents 已 detach，不再携带 new_vae 的梯度 ----
            with make_ctx_g():
                new_lq_latents_detached = new_lq_latents.detach()

                # gen_start_point / fixed_timestep / one_step_sigma 已在预处理阶段计算，直接复用
                random_timestep_id_for_dis = torch.randint(0, _safe_unwrap(discriminator).pipe.scheduler.num_train_timesteps, (1,))
                random_timestep = _safe_unwrap(discriminator).pipe.scheduler.timesteps[random_timestep_id_for_dis].to(device=accelerator.device)

                noise = torch.randn_like(new_lq_latents_detached)
                noisy_latents = _safe_unwrap(generator).pipe.scheduler.add_noise(new_lq_latents_detached, noise, fixed_timestep)



                noise_pred = generator(
                    noisy_latents,
                    edit_latents,
                    fixed_timestep,
                    **pre_saved_prompt_emb,edit_masks_parts=edit_masks_parts
                ).to(dtype=torch.bfloat16)

                training_pred = noisy_latents + (0 - one_step_sigma) * noise_pred

                # ---- Loss 计算 ----
                training_pred_rgb = _safe_unwrap(generator).pipe.vae.decode(training_pred)

                loss_rgb_mse = pixel_loss_fn(training_pred_rgb.float(), gt_rgb.detach().float())


                use_mask_loss = True
                if use_mask_loss and lq_mask is not None:
                    loss_mask_mse = torch.tensor(0.0, device=accelerator.device)
                    # 算 Unreduced MSE (保留每个像素的独立误差)
                    mse_unreduced = torch.nn.functional.mse_loss(
                        training_pred_rgb.float(), 
                        gt_rgb.detach().float(), 
                        reduction='none'
                    )
    
                    # 确保 Mask 在 RGB 尺寸下 (B, 1, H, W)
                    if lq_mask.dim() == 3:
                        lq_mask_4d = lq_mask.unsqueeze(0).float()
                    elif lq_mask.dim() == 2:
                        lq_mask_4d = lq_mask.unsqueeze(0).unsqueeze(0).float()
                    else:
                        lq_mask_4d = lq_mask.float()
                    lq_mask_rgb = torch.nn.functional.interpolate(lq_mask_4d, size=training_pred_rgb.shape[2:], mode='nearest')

                    # 【核心逻辑】：只求 Mask 内部的均值！
                    # sum(误差 * mask) / sum(mask面积 + 1e-8防止除零)
                    loss_mask_mse = (mse_unreduced * lq_mask_rgb).sum() / (lq_mask_rgb.sum() + 1e-8)

                    # 3. 汇总加权
                    # rgb_w 是全局基础权重，mask_w_ratio 是你给重点区域额外施加的权重（比如 1.0 或更高）
                    mask_w_ratio = 1.5
                    print(f"loss_rgb_mse: {loss_rgb_mse.item()}, loss_mask_mse: {loss_mask_mse.item()}")
                    loss_rgb_mse = (rgb_w * loss_rgb_mse) + (mask_w_ratio * loss_mask_mse)

                else:
                    loss_rgb_mse = rgb_w * loss_rgb_mse

                loss_lpips = net_lpips(training_pred_rgb.float(), gt_rgb.detach().float()).mean()
                loss_lpips = lpips_w * loss_lpips

                loss_dists = torch.tensor(0.0, device=accelerator.device)
                if net_dists is not None:
                    training_pred_rgb_norm = ((training_pred_rgb.float() + 1.0) / 2.0).clamp(0.0, 1.0)
                    gt_rgb_norm = ((gt_rgb.detach().float() + 1.0) / 2.0).clamp(0.0, 1.0)
                    loss_dists = dists_w * net_dists(training_pred_rgb_norm, gt_rgb_norm).mean()
                    print(f"loss_dists: {loss_dists.item()}")

                # 判别器条件
                if ref_latents is not None:
                    d_condition_latents = ref_latents
                else:
                    d_condition_latents = lq_latents

                # GAN loss for G
                with torch.no_grad():
                    real_d_pred = discriminator(
                        deal_discriminator_condition(gt_latents, d_condition_latents, dataset_yaml['use_dual_condition_flag']),
                        random_timestep, prompt_emb
                    ).detach().clone()
                fake_g_pred = discriminator(
                    deal_discriminator_condition(training_pred, d_condition_latents, dataset_yaml['use_dual_condition_flag']),
                    random_timestep, prompt_emb
                )
                tmp1 = cri_gan(
                    real_d_pred - adaptive_relaxed_mean(fake_g_pred, output_size=dataset_yaml['relaxed_mean_size']),
                    False, is_disc=False
                )
                tmp2 = cri_gan(
                    fake_g_pred - adaptive_relaxed_mean(real_d_pred, output_size=dataset_yaml['relaxed_mean_size']),
                    True, is_disc=False
                )
                loss_g_gan = (tmp1 + tmp2) / 2

                total_loss = (loss_rgb_mse + loss_lpips + loss_dists + loss_g_gan * gan_w_ratio)
                total_loss = total_loss / gradient_accumulation_steps
                accelerator.backward(total_loss)

            if sync_gradients:
                if args.clip_grad_norm:
                    accelerator.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_g.step()
                optimizer_g.zero_grad()
            

            # =====================================================================
            # Train D（与 train_gan.py 完全一致）
            # =====================================================================
            for name, param in discriminator.named_parameters():
                if name in dis_trainable_param_names:
                    param.requires_grad = True

            ctx_d = accelerator.no_sync(discriminator) if not sync_gradients else nullcontext()

            with ctx_d:
                fake_d_pred = fake_g_pred.detach().clone()
                real_d_pred = discriminator(
                    deal_discriminator_condition(gt_latents, d_condition_latents, dataset_yaml['use_dual_condition_flag']),
                    random_timestep, prompt_emb
                )
                tmp_3 = cri_gan(
                    real_d_pred - adaptive_relaxed_mean(fake_d_pred, output_size=dataset_yaml['relaxed_mean_size']),
                    True, is_disc=True
                )

                # R1 loss
                noised_gt_latents = add_gaussian_noise(gt_latents, variance=dataset_yaml['variance'])
                real_d_pred_R1 = discriminator(
                    deal_discriminator_condition(noised_gt_latents, d_condition_latents, dataset_yaml['use_dual_condition_flag']),
                    random_timestep, prompt_emb
                )
                tmp_4 = torch.nn.functional.mse_loss(real_d_pred.float(), real_d_pred_R1.float())
                tmp_4 = dataset_yaml['r1_regularization'] * tmp_4

                fake_d_pred = discriminator(
                    deal_discriminator_condition(training_pred.detach(), d_condition_latents, dataset_yaml['use_dual_condition_flag']),
                    random_timestep, prompt_emb
                )
                tmp_5 = cri_gan(
                    fake_d_pred - adaptive_relaxed_mean(real_d_pred.detach(), output_size=dataset_yaml['relaxed_mean_size']),
                    False, is_disc=True
                )
                tmp_6 = 0

                accelerator.backward((tmp_3 * 0.5 + tmp_4 + tmp_5 * 0.5 + tmp_6) / gradient_accumulation_steps)

            if accelerator.is_main_process:
                if iteration % dataset_yaml['viz_iters'] == 1:
                    with torch.no_grad():
                        print(real_d_pred - adaptive_relaxed_mean(fake_d_pred, output_size=dataset_yaml['relaxed_mean_size']))

            if sync_gradients:
                if args.clip_grad_norm:
                    accelerator.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                optimizer_d.step()
                optimizer_d.zero_grad()

                # 可选：更新学习率（防止超过总步数后 cosine 回升）
                if scheduler_g is not None and iteration < total_training_steps:
                    scheduler_g.step()
                if scheduler_d is not None and iteration < total_training_steps:
                    scheduler_d.step()

            # ---- logging ----
            if accelerator.is_main_process:
                log_dict = {
                    "loss_rgb_mse": loss_rgb_mse.item(),
                    "loss_lpips": loss_lpips.item(),
                    "loss_g_gan_tmp1": tmp1.item(),
                    "loss_g_gan_tmp2": tmp2.item(),
                    "loss_new_vae_lq": loss_new_vae_lq.item(),
                    "loss_d_real": tmp_3.item(),
                    "loss_d_real_r1": tmp_4.item() if torch.is_tensor(tmp_4) else tmp_4,
                    "loss_d_fake": tmp_5.item(),
                    "loss_d_fake_r2": tmp_6.item() if torch.is_tensor(tmp_6) else tmp_6,
                    "loss_dists": loss_dists.item() if torch.is_tensor(loss_dists) else loss_dists,
                }
                if scheduler_g is not None:
                    log_dict["lr_g"] = optimizer_g.param_groups[0]['lr']
                    log_dict["lr_d"] = optimizer_d.param_groups[0]['lr']
                if trt_logger is not None:
                    trt_logger.log_metrics(log_dict, step=log_iteration)
                log_iteration += 1

            # ---- viz & ckpt ----
            if sync_gradients:
                iteration += 1

                if accelerator.is_main_process:
                    if iteration % dataset_yaml['viz_iters'] == 1:
                        os.makedirs(os.path.join(workdir, "viz"), exist_ok=True)
                        print(caption)

                        with torch.no_grad():
                            res_img = _safe_unwrap(generator).pipe.vae_output_to_image(training_pred_rgb)
                            res_img2 = _safe_unwrap(generator).pipe.vae_output_to_image(new_lq_latents_rgb)

                        imgs = [gt_pil, lq_pil, res_img, res_img2]
                        lq_for_cond_pil = data.get("lq_for_cond", None)
                        if lq_for_cond_pil is not None:
                            imgs.append(lq_for_cond_pil)
                        if full_lq_pil is not None:
                            imgs.append(full_lq_pil)
                        if ref_pil is not None:
                            imgs.append(ref_pil)

                        target_h = gt_pil.height
                        resized = []
                        for im in imgs:
                            if im.height != target_h:
                                scale_factor = target_h / im.height
                                im = im.resize((int(im.width * scale_factor), target_h), Image.BICUBIC)
                            resized.append(im)

                        total_w = sum(im.width for im in resized)
                        canvas = Image.new('RGB', (total_w, target_h))
                        x_offset = 0
                        for im in resized:
                            canvas.paste(im, (x_offset, 0))
                            x_offset += im.width
                        canvas.save(os.path.join(workdir, "viz", f"iter_{iteration}.png"))

                        # save prompt
                        with open(os.path.join(workdir, "viz", f"iter_{iteration}.txt"), "w", encoding='utf-8') as f:
                            f.write(prompt_text)

                    if iteration % dataset_yaml['save_ckpt_iters'] == 1:
                        _safe_unwrap(generator).save_ckpt(
                            os.path.join(workdir, 'checkpoints'), iter=iteration, tag="gen"
                        )
                        _safe_unwrap(discriminator).save_ckpt(
                            os.path.join(workdir, 'checkpoints'), iter=iteration, tag="dis"
                        )

        # ---- epoch 结束后的测试 ----
        if args.test_lq_txt:
            accelerator.wait_for_everyone()
            accelerator.print(f"Starting distributed test after epoch {epoch}...")
            try:
                distributed_test(
                    accelerator, generator, epoch, workdir,
                    test_lq_txt=args.test_lq_txt,
                    test_ref_txt=args.test_ref_txt,
                    test_prompt_txt=args.test_prompt_txt,
                    test_gt_txt=args.test_gt_txt,
                    test_scale=args.test_scale,
                    test_cfg=dataset_yaml.get('test_cfg', 1.0),
                    test_metrics=dataset_yaml.get('test_metrics', ''),
                    test_crop_border=dataset_yaml.get('test_crop_border', 0),
                    use_full_lq_condition=dataset_yaml.get('use_full_lq_condition', False),
                    ref_max_pixels=dataset_yaml.get('ref_max_pixels', None),
                    full_lq_max_pixels=dataset_yaml.get('full_lq_max_pixels', None),
                    drop_lq_crop_condition=dataset_yaml.get('lq_dropout_prob', 0.0) >= 1.0,
                )
            except Exception as e:
                accelerator.print(f"[Warning] Distributed test failed at epoch {epoch}: {e}")
                import traceback
                accelerator.print(traceback.format_exc())
            finally:
                gen_module = accelerator.unwrap_model(generator)
                gen_module.train()
                accelerator.wait_for_everyone()


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        raise NotImplementedError("")
    elif args.task == "train":
        train(args)
