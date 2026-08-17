"""
基于原生 Qwen-Image-Edit 的 Generator。

与原版 generator.py 的核心区别：
- 使用原生 model_fn_qwen_image（不替换为 model_fn_qwen_image_sr）
- 使用原生 QwenImageUnit_EditImageEmbedder（Pipeline 默认自带，不手动插入）
- 使用标准 LoRA（不使用 DualLoRA/TriLoRA）
- 使用 zero_cond_t 区分条件/噪声（不使用 ConditionTypeEmbedding）
- 条件图不需要手动加噪（zero_cond_t 自动给条件图 timestep=0）
"""
import torch
import torch.nn as nn
import torchvision
import numpy as np
from einops import rearrange
import json
from copy import deepcopy
from PIL import Image

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from base import BaseModelForT2ILoRA


class StandardLoRALinear(torch.nn.Module):
    """标准 LoRA 层：冻结原始权重，只训练低秩分解矩阵 A/B"""

    def __init__(self, original_linear, rank, alpha=None):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_down = torch.nn.Linear(in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, out_features, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight)
        torch.nn.init.zeros_(self.lora_up.weight)

        # 将 LoRA 权重的 dtype 对齐到原始层，避免 bf16 输入与 fp32 权重不匹配
        original_dtype = original_linear.weight.dtype
        self.lora_down = self.lora_down.to(dtype=original_dtype)
        self.lora_up = self.lora_up.to(dtype=original_dtype)

        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base_output = self.original_linear(x)
        lora_output = self.lora_up(self.lora_down(x)) * self.scaling
        return base_output + lora_output


def replace_linear_with_standard_lora(model, target_patterns, rank, alpha=None):
    """将模型中匹配 target_patterns 的 Linear 层替换为 StandardLoRALinear"""
    # 先收集所有 (parent, attr_name, module) 再替换，避免迭代中修改
    modules_to_replace = []
    named_modules_dict = dict(model.named_modules())

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not any(pattern in name for pattern in target_patterns):
            continue
        # 如果已经是 LoRA 层则跳过
        if isinstance(module, StandardLoRALinear):
            continue

        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = named_modules_dict[parent_name]
        else:
            attr_name = parts[0]
            parent = model

        modules_to_replace.append((parent, attr_name, module))

    for parent, attr_name, module in modules_to_replace:
        lora_layer = StandardLoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, attr_name, lora_layer)

    return len(modules_to_replace)


class GeneratorEdit(BaseModelForT2ILoRA):
    """
    基于原生 Qwen-Image-Edit 的 Generator。

    条件图（LQ / REF）通过原生 edit_latents 参数注入 model_fn_qwen_image，
    用 zero_cond_t 机制区分噪声流（timestep=t）和条件流（timestep=0），
    使用标准 LoRA 微调 DiT。
    """

    def __init__(
        self,
        torch_dtype=torch.float16,
        pretrained_weights=None,
        tokenizer_path=None,
        processor_path=None,
        learning_rate=1e-4,
        use_gradient_checkpointing=True,
        pretrained_ckpt_path_gen=None,
        gen_start_point=750,
        train_new_vae=True,
        lora_rank=128,
        lora_alpha=None,
        lora_target_modules="to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.0.proj,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1",
        zero_cond_t=True,
    ):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        self.gen_start_point = gen_start_point
        self.zero_cond_t = zero_cond_t

        # ---- 加载 Pipeline ----
        model_configs = []
        if pretrained_weights is not None:
            pretrained_weights = json.loads(pretrained_weights)
            for path in pretrained_weights:
                model_configs.append(ModelConfig(path=path))

        tokenizer_config = (
            ModelConfig(tokenizer_path)
            if tokenizer_path
            else ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/")
        )
        processor_config = (
            ModelConfig(processor_path)
            if processor_path
            else ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/")
        )

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            processor_config=processor_config,
        )

        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.pipe.freeze_except([])

        self.mask_in = nn.Linear(4, 64)

        # 【零初始化】
        nn.init.zeros_(self.mask_in.weight)
        nn.init.zeros_(self.mask_in.bias)
        self.mask_in = self.mask_in.to(dtype=torch.bfloat16)

        # 不做任何 model_fn 替换 —— Pipeline 默认使用 model_fn_qwen_image
        # 不手动插入任何自定义 Unit —— Pipeline 默认自带 QwenImageUnit_EditImageEmbedder

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # ---- new_vae（可选，用于学习 LQ→GT latent 映射）----
        self.pipe.new_vae = deepcopy(self.pipe.vae)
        if train_new_vae:
            self._unfrozen_conv_layers(self.pipe.new_vae.encoder, type(self.pipe.new_vae.encoder.conv_in))

        # ---- 标准 LoRA 注入 ----
        target_patterns = [m.strip() for m in lora_target_modules.split(",")]
        replaced_count = replace_linear_with_standard_lora(
            self.pipe.dit,
            target_patterns=target_patterns,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        print(f"[GeneratorEdit] Replaced {replaced_count} Linear layers with StandardLoRA (rank={lora_rank})")

        # ---- 加载预训练 checkpoint ----
        if pretrained_ckpt_path_gen:
            ckpt_state_dict = torch.load(pretrained_ckpt_path_gen, map_location="cpu")
            self.load_state_dict(ckpt_state_dict, strict=False)

    def _unfrozen_conv_layers(self, model, target_cls):
        """解冻 new_vae encoder 中指定类型的卷积层"""
        for name, module in model.named_modules():
            if isinstance(module, target_cls) and "time_conv" not in name:
                for param in module.parameters():
                    param.requires_grad = True

    def configure_optimizers(self):
        trainable_modules = list(filter(lambda p: p.requires_grad, self.pipe.parameters()))
        # 将 mask_in 参数加入可训练列表
        trainable_modules += list(self.mask_in.parameters())
        optimizer = torch.optim.RMSprop(
            trainable_modules,
            lr=self.learning_rate,
            alpha=0.9,
            momentum=0.0,
        )
        return optimizer

    def forward(self, noisy_latents, edit_latents, timestep, prompt_emb, prompt_emb_mask, edit_masks_parts=None):
        """
        前向传播 — 调用原生 model_fn_qwen_image。

        Args:
            noisy_latents: [B, C, H, W] 加噪的目标 latents
            edit_latents:  条件图 latents，支持：
                           - 单张 [B, C, H, W]
                           - 列表 [lq_latent, ref_latent]（原生支持不同分辨率）
            timestep:      时间步 tensor
            prompt_emb:    文本嵌入
            prompt_emb_mask: 文本掩码
        """
        batch_size, channels, height_latent, width_latent = noisy_latents.shape
        # process ref mask to ref latents
        if edit_latents is not None and edit_masks_parts is not None:
            edit_image = []
            edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
            for i, (e, current_mask) in enumerate(zip(edit_latents_list, edit_masks_parts)):
                if current_mask is not None:
                    batch_size = e.shape[0]
                    # ref_mask: [1, H, W] -> [B, 1, H, W]
                    current_mask = current_mask.to(dtype=self.mask_in.weight.dtype)
                    current_mask = current_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
                    # 展平 Mask (注意 C=1)
                    mask_tokens = rearrange(current_mask, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2, C=1)
                    # 映射并相加 (因为 zero-init，开始时对 tokens 毫无影响)
                    mask_condition = self.mask_in(mask_tokens)
                    mask_condition_2d = rearrange(mask_condition, "B (H W) (C P Q) -> B C (H P) (W Q)", 
                            H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2, C=16)
                    # 提前融合ref latents和ref mask
                    e = e + mask_condition_2d
                edit_image.append(e)
            if len(edit_image) == 0:
                edit_latents = None
            elif len(edit_image) == 1:
                edit_latents = edit_image[0]
            else:
                edit_latents = edit_image
        out = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=noisy_latents,
            timestep=timestep,
            prompt_emb=prompt_emb,
            prompt_emb_mask=prompt_emb_mask,
            height=height_latent * 8,
            width=width_latent * 8,
            edit_latents=edit_latents,
            zero_cond_t=self.zero_cond_t,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        return out

    def _find_prompt_embedder(self):
        """从 pipeline units 中找到 PromptEmbedder unit"""
        for unit in self.pipe.units:
            if hasattr(unit, "encode_prompt"):
                return unit
        raise RuntimeError("Cannot find PromptEmbedder unit in pipeline")

    def encode_prompt(self, prompt, edit_image=None):
        """
        编码文本 prompt。

        Args:
            prompt: 文本提示
            edit_image: 条件图（PIL Image 或列表），若提供则走 edit 模式的 prompt 模板
        Returns:
            dict: {'prompt_emb': tensor, 'prompt_emb_mask': tensor}
        """
        prompt_embedder = self._find_prompt_embedder()
        return prompt_embedder.process(self.pipe, prompt, edit_image=edit_image)

    def encode_image_to_latent(self, pil_image):
        """将 PIL Image 编码为 VAE latent 和 RGB tensor"""
        rgb = self.pipe.preprocess_image(pil_image).to(
            device=self.pipe.device, dtype=self.pipe.torch_dtype
        )
        latent = self.pipe.vae.encode(rgb, tiled=False)
        return latent, rgb

    @torch.no_grad()
    def infer(
        self,
        prompt,
        negative_prompt,
        edit_image,
        cfg_scale=4.0,
        num_inference_steps=30,
        seed=42,
        height=None,
        width=None,
    ):
        """
        推理 — 使用原生 Pipeline 的完整多步去噪。

        Args:
            prompt: 文本提示
            negative_prompt: 负面提示
            edit_image: 条件图。可以是单张 PIL Image 或列表 [lq_pil, ref_pil]
            cfg_scale: Classifier-Free Guidance 强度
            num_inference_steps: 去噪步数
            seed: 随机种子
            height: 输出高度（默认使用条件图高度）
            width: 输出宽度（默认使用条件图宽度）
        """
        if isinstance(edit_image, list):
            ref_img = edit_image[0]
        else:
            ref_img = edit_image

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            edit_image=edit_image,
            edit_image_auto_resize=True,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            height=height or ref_img.size[1],
            width=width or ref_img.size[0],
            zero_cond_t=self.zero_cond_t,
        )
        return result

    @torch.no_grad()
    def infer_one_step(
        self,
        prompt,
        negative_prompt,
        lq_pil,
        ref_pil=None,
        full_lq_pil=None,
        cfg_scale=1.0,
        tiled=False,
        tile_size=64,
        tile_stride=48,
        tile_prompt_mode="global",
        fidelity=1.0,
        fidelity_mask=None,
        drop_lq_crop_condition=False,
        lq_mask_pil=None,           # 新增：LQ mask, PIL L mode, 白=需要ref
        ref_mask_pil=None,          # 新增：REF mask, PIL L mode, 白=有效纹理
        mask_low_noise_ratio=0.1,   # 新增：mask=0 区域噪声衰减比例
        inject_mask_latent=True,    # 新增：是否将 mask latent 注入 edit_latents（False=仅噪声调制）
        post_decode_fn=None,        # 可选：vae.decode 后、转 PIL 前的回调函数，接收 tensor [-1,1] 返回 tensor [-1,1]
    ):
        """
        单步推理（保留与 GAN 训练兼容的一步去噪推理方式）。

        Args:
            prompt: 文本提示
            negative_prompt: 负面提示
            lq_pil: 低质量输入图（PIL Image）
            ref_pil: 参考图（PIL Image，可选）
            full_lq_pil: 全局 LQ 条件图（PIL Image，可选）。
                传入即启用三条件模式 [lq_crop, full_lq, ref]。
                tiled 模式下 lq_latents 在 [0] 会被按 tile crop，full_lq 保持全局提供完整上下文。
                可通过像素预算限制独立控制其分辨率。若为 None 则不启用三条件。
            cfg_scale: CFG 强度
            tiled: 是否启用分块推理（大图必须开启，否则显存不足）
            tile_size: latent 空间的 tile 大小（像素空间为 tile_size * 8）
            tile_stride: latent 空间的 tile 步长（< tile_size 则有 overlap）
            tile_prompt_mode: prompt 编码模式，仅在 tiled=True 时生效。
                "global" — 所有 tile 共享整张图编码的 prompt（快，但与训练不一致）
                "tile"   — 为每个 tile 裁剪对应区域的 LQ 独立编码 prompt（慢，与训练一致）
        """
        # 编码条件图（大图用 tiled encode 避免 VAE 爆显存）
        lq_rgb = self.pipe.preprocess_image(lq_pil).to(
            device=self.device, dtype=self.pipe.torch_dtype
        )
        vae_tiled_kwargs = dict(tiled=tiled, tile_size=tile_size * 8, tile_stride=tile_stride * 8) if tiled else dict(tiled=False)
        new_lq_latents = self.pipe.new_vae.encode(lq_rgb, **vae_tiled_kwargs)
        lq_latents = self.pipe.vae.encode(lq_rgb, **vae_tiled_kwargs)

        # 编码 full_lq 条件图（如果提供，启用三条件模式）
        full_lq_latents = None
        if full_lq_pil is not None:
            full_lq_w, full_lq_h = full_lq_pil.size
            if full_lq_w % 16 != 0 or full_lq_h % 16 != 0:
                aligned_w = max(16, full_lq_w // 16 * 16)
                aligned_h = max(16, full_lq_h // 16 * 16)
                print(f"[GeneratorEdit] Warning: full_lq_pil size ({full_lq_w}, {full_lq_h}) is not a multiple of 16, "
                      f"auto-aligning to ({aligned_w}, {aligned_h}).")
                full_lq_pil = full_lq_pil.resize((aligned_w, aligned_h), Image.LANCZOS)
            full_lq_rgb = self.pipe.preprocess_image(full_lq_pil).to(
                device=self.device, dtype=self.pipe.torch_dtype
            )
            full_lq_latents = self.pipe.vae.encode(full_lq_rgb, tiled=False)

        if ref_pil is not None:
            # 校验 REF 尺寸必须是 16 的倍数（调用方应提前对齐），
            # 否则 VAE 编码后 latent 可能出现奇数维度导致 rearrange 报错
            ref_w, ref_h = ref_pil.size
            if ref_w % 16 != 0 or ref_h % 16 != 0:
                aligned_w = max(16, ref_w // 16 * 16)
                aligned_h = max(16, ref_h // 16 * 16)
                print(f"[GeneratorEdit] Warning: ref_pil size ({ref_w}, {ref_h}) is not a multiple of 16, "
                      f"auto-aligning to ({aligned_w}, {aligned_h}). "
                      f"Consider aligning in the caller to avoid extra interpolation.")
                ref_pil = ref_pil.resize((aligned_w, aligned_h), Image.LANCZOS)
                ref_mask_pil = ref_mask_pil.resize((aligned_w, aligned_h), Image.Resampling.NEAREST)

            ref_rgb = self.pipe.preprocess_image(ref_pil).to(
                device=self.device, dtype=self.pipe.torch_dtype
            )
            ref_latents = self.pipe.vae.encode(ref_rgb, tiled=False)
        else:
            ref_latents = None
        # 提前判断是否启用 SAM mask 噪声调制 / mask latent 注入
        # use_sam_mask = (lq_mask_pil is not None and ref_mask_pil is not None and ref_latents is not None)
        use_sam_mask = False
        use_mask_latent_injection = use_sam_mask and inject_mask_latent

        # 编码全局 prompt（global 模式直接使用；tile 模式下也作为 fallback / non-tiled 路径使用）
        # edit_image 数量和顺序必须与 edit_latents 一致
        global_edit_image_parts = [] if drop_lq_crop_condition else [lq_pil]
        if full_lq_pil is not None:
            global_edit_image_parts.append(full_lq_pil)
        if ref_pil is not None:
            global_edit_image_parts.append(ref_pil)
        # ===== 新增：LQ mask 加入 prompt_edit_images =====
        if lq_mask_pil is not None:
            global_edit_image_parts.append(lq_mask_pil.convert("RGB"))
        # ===== 新增结束 =====
        global_edit_image = global_edit_image_parts if len(global_edit_image_parts) > 1 else global_edit_image_parts[0]
        posi_prompt_result = self.encode_prompt(prompt, edit_image=global_edit_image)
        posi_prompt_emb = {
            "prompt_emb": posi_prompt_result["prompt_emb"],
            "prompt_emb_mask": posi_prompt_result["prompt_emb_mask"],
        }

        # 生成 per-tile prompt（仅 tiled + tile 模式）
        tile_posi_prompt_embs = None
        tile_posi_prompt_emb_masks = None
        tile_nega_prompt_embs = None
        tile_nega_prompt_emb_masks = None

        if tiled and tile_prompt_mode == "tile":
            tile_posi_prompt_embs, tile_posi_prompt_emb_masks = self._encode_tile_prompts(
                prompt, lq_pil, ref_pil, lq_latents, tile_size, tile_stride,
                full_lq_pil=full_lq_pil,
            )
            if cfg_scale != 1.0:
                tile_nega_prompt_embs, tile_nega_prompt_emb_masks = self._encode_tile_prompts(
                    negative_prompt, lq_pil, ref_pil, lq_latents, tile_size, tile_stride,
                    full_lq_pil=full_lq_pil,
                )

        # 加噪
        start_timestep = self.gen_start_point
        fixed_timestep_id = torch.tensor([start_timestep])
        fixed_timestep = self.pipe.scheduler.timesteps[fixed_timestep_id].to(device=self.device)
        one_step_sigma = self.pipe.scheduler.sigmas[fixed_timestep_id].to(
            dtype=torch.bfloat16, device=self.device
        )

        noise = torch.randn_like(new_lq_latents)

        # ======== SAM Mask 噪声调制 + Mask 条件注入 ========
        mask_latent = None
        ref_mask_latent = None

        if use_sam_mask:
            _, _, h_lat, w_lat = lq_latents.shape
            lq_mask_resized = lq_mask_pil.convert('L').resize((w_lat, h_lat), Image.BILINEAR)
            lq_mask_tensor = torch.tensor(
                list(lq_mask_resized.getdata()), dtype=torch.float32
            ).reshape(1, 1, h_lat, w_lat).to(device=self.device, dtype=self.pipe.torch_dtype) / 255.0

            # Per-pixel 噪声调制
            start_sigma = self.pipe.scheduler.sigmas[torch.tensor([start_timestep])].to(
                device=self.device, dtype=torch.float32
            )
            per_pixel_sigma = start_sigma * (
                lq_mask_tensor.float() + mask_low_noise_ratio * (1 - lq_mask_tensor.float())
            )
            per_pixel_sigma = per_pixel_sigma.to(dtype=lq_latents.dtype)
            lq_latents = (1 - per_pixel_sigma) * lq_latents.detach() + per_pixel_sigma * noise

            # Mask condition latent（仅在 inject_mask_latent=True 时注入 edit_latents）
            if inject_mask_latent:
                mask_latent = lq_mask_tensor.expand(-1, 16, -1, -1)

                _, _, h_ref, w_ref = ref_latents.shape
                ref_mask_resized = ref_mask_pil.convert('L').resize((w_ref, h_ref), Image.BILINEAR)
                ref_mask_tensor = torch.tensor(
                    list(ref_mask_resized.getdata()), dtype=torch.float32
                ).reshape(1, 1, h_ref, w_ref).to(device=self.device, dtype=self.pipe.torch_dtype) / 255.0
                ref_mask_latent = ref_mask_tensor.expand(-1, 16, -1, -1)

        # fidelity：对条件 lq_latents 加噪，降低条件约束强度
        # fidelity=1.0（默认）不加噪；fidelity 越小条件约束越强
        elif fidelity_mask is not None:
            # --- 基于 mask 的逐像素 fidelity ---
            # mask=0(黑) → 使用 fidelity 参数值；mask=255(白) → fidelity=1.0(不加噪)；中间线性插值
            _, _, h_lat, w_lat = lq_latents.shape
            mask_pil = fidelity_mask.convert('L').resize((w_lat, h_lat), Image.BILINEAR)
            mask_tensor = torch.tensor(
                list(mask_pil.getdata()), dtype=torch.float32
            ).reshape(1, 1, h_lat, w_lat).to(device=self.device) / 255.0  # [0, 1]

            # 逐像素 fidelity 值：mask=0 → fidelity, mask=1 → 1.0
            per_pixel_fidelity = fidelity + mask_tensor * (1.0 - fidelity)  # shape: [1,1,H,W]

            # 对应的 timestep_id（浮点）
            per_pixel_timestep_id = (
                start_timestep + per_pixel_fidelity * (1000 - start_timestep)
            ).clamp(0, 999)

            # 获取每个像素对应的 sigma 值（线性插值）
            all_sigmas = self.pipe.scheduler.sigmas.to(device=self.device, dtype=torch.float32)
            floor_ids = per_pixel_timestep_id.long().clamp(0, 998)
            ceil_ids = (floor_ids + 1).clamp(0, 999)
            frac = per_pixel_timestep_id - floor_ids.float()
            sigma_floor = all_sigmas[floor_ids.reshape(-1)].reshape(1, 1, h_lat, w_lat)
            sigma_ceil = all_sigmas[ceil_ids.reshape(-1)].reshape(1, 1, h_lat, w_lat)
            per_pixel_sigma = (sigma_floor * (1 - frac) + sigma_ceil * frac).to(dtype=lq_latents.dtype)

            # flow matching 加噪：noisy = (1 - sigma) * latents + sigma * noise
            no_noise_mask = (per_pixel_fidelity >= 1.0 - 1e-6)
            lq_latents_noisy = (1 - per_pixel_sigma) * lq_latents.detach() + per_pixel_sigma * noise
            lq_latents = torch.where(no_noise_mask, lq_latents.detach(), lq_latents_noisy)
        else:
            # --- 原有标量 fidelity 逻辑 ---
            fidelity_timestep_id = int(start_timestep + fidelity * (1000 - start_timestep) + 0.5)
            if fidelity_timestep_id == 1000:
                pass
            else:
                fidelity_timestep_id = torch.randint(fidelity_timestep_id, fidelity_timestep_id + 1, (1,))
                fidelity_timestep = self.pipe.scheduler.timesteps[fidelity_timestep_id].to(device=self.device)
                lq_latents = self.pipe.scheduler.add_noise(lq_latents.detach(), noise, fidelity_timestep)

        # ======== Mask 下采样条件注入 ========
        def downsample_mask(mask_pil, device):
            if mask_pil is not None:
                w, h = mask_pil.size
                resized_mask = mask_pil.resize((w//8, h//8), Image.Resampling.BOX)
                arr = np.array(resized_mask)
                arr = (arr > 0).astype(np.uint8) * 255
                resized_mask = Image.fromarray(arr, 'L')
                mask_tensor = (torchvision.transforms.ToTensor()(resized_mask)>0).to(device)
                return mask_tensor
            else:
                return None

        lq_mask = downsample_mask(lq_mask_pil, self.device)
        ref_mask = downsample_mask(ref_mask_pil, self.device)

        # 组装 edit_latents
        # tiled 模式下 [0] 会被按 tile crop，其余保持全局
        # drop_lq_crop=True 时跳过 lq_crop，只保留 full_lq 和 ref
        edit_latents_parts = [] if drop_lq_crop_condition else [lq_latents]
        if full_lq_latents is not None:
            edit_latents_parts.append(full_lq_latents)
        if ref_latents is not None:
            if ref_mask is not None:
                batch_size = ref_latents.shape[0]
                # ref_mask: [1, H, W] -> [B, 1, H, W]
                ref_mask = ref_mask.to(dtype=self.mask_in.weight.dtype)
                ref_mask = ref_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
                # 展平 Mask (注意 C=1)
                mask_tokens = rearrange(ref_mask, "B C (H P) (W Q) -> B (H W) (C P Q)", H=ref_latents.shape[2]//2, W=ref_latents.shape[3]//2, P=2, Q=2, C=1)
                # 映射并相加 (因为 zero-init，开始时对 tokens 毫无影响)
                mask_condition = self.mask_in(mask_tokens)
                mask_condition_2d = rearrange(mask_condition, "B (H W) (C P Q) -> B C (H P) (W Q)", 
                    H=ref_latents.shape[2]//2, W=ref_latents.shape[3]//2, P=2, Q=2, C=16)
                # 提前融合ref latents和ref mask
                ref_latents = ref_latents + mask_condition_2d
            edit_latents_parts.append(ref_latents)

        if len(edit_latents_parts) == 1:
            edit_latents = edit_latents_parts[0]
        else:
            edit_latents = edit_latents_parts

        noisy_latents = self.pipe.scheduler.add_noise(new_lq_latents.detach(), noise, fixed_timestep)
        batch_size, channels, height_latent, width_latent = noisy_latents.shape

        # 构建 model_fn 的公共参数
        model_common_kwargs = dict(
            dit=self.pipe.dit,
            timestep=fixed_timestep,
            height=height_latent * 8,
            width=width_latent * 8,
            edit_latents=edit_latents,
            zero_cond_t=self.zero_cond_t,
        )

        # 前向（正向 prompt）
        if tiled:
            from diffsynth.pipelines.qwen_image import tiled_model_fn_qwen_image
            noise_pred_posi = tiled_model_fn_qwen_image(
                latents=noisy_latents,
                tile_size=tile_size,
                tile_stride=tile_stride,
                tile_prompt_embs=tile_posi_prompt_embs,
                tile_prompt_emb_masks=tile_posi_prompt_emb_masks,
                all_edit_latents_global=drop_lq_crop_condition,
                **posi_prompt_emb,
                **model_common_kwargs,
            )
        else:
            noise_pred_posi = self.pipe.model_fn(
                latents=noisy_latents,
                **posi_prompt_emb,
                **model_common_kwargs,
            )

        # CFG（如果需要）
        if cfg_scale != 1.0:
            nega_prompt_result = self.encode_prompt(negative_prompt, edit_image=global_edit_image)
            nega_prompt_emb = {
                "prompt_emb": nega_prompt_result["prompt_emb"],
                "prompt_emb_mask": nega_prompt_result["prompt_emb_mask"],
            }
            if tiled:
                noise_pred_nega = tiled_model_fn_qwen_image(
                    latents=noisy_latents,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                    tile_prompt_embs=tile_nega_prompt_embs,
                    tile_prompt_emb_masks=tile_nega_prompt_emb_masks,
                    all_edit_latents_global=drop_lq_crop_condition,
                    **nega_prompt_emb,
                    **model_common_kwargs,
                )
            else:
                noise_pred_nega = self.pipe.model_fn(
                    latents=noisy_latents,
                    **nega_prompt_emb,
                    **model_common_kwargs,
                )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        # 一步预测
        training_pred = noisy_latents + (0 - one_step_sigma) * noise_pred

        # 解码（大图用 tiled decode 避免 VAE 爆显存）
        image = self.pipe.vae.decode(training_pred, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # Apply post-decode function (e.g., NaturalizationRefineHead) before converting to PIL
        if post_decode_fn is not None:
            image = post_decode_fn(image.float())

        image = self.pipe.vae_output_to_image(image)
        return image

    def _encode_tile_prompts(self, prompt, lq_pil, ref_pil, lq_latents, tile_size, tile_stride,
                             full_lq_pil=None):
        """
        为每个 tile 裁剪对应区域的 LQ 图片，独立编码 prompt。
        遍历顺序与 tiled_model_fn_qwen_image 完全一致（行优先），保证 index 对齐。

        Args:
            prompt: 文本提示
            lq_pil: 完整 LQ 图片（PIL Image）
            ref_pil: 参考图（PIL Image 或 None），不裁剪，保持全局
            lq_latents: LQ latent tensor，用于获取 latent 空间尺寸
            tile_size: latent 空间 tile 大小
            tile_stride: latent 空间 tile 步长
            full_lq_pil: 全局 LQ 条件图（PIL Image，可选），传入即启用三条件模式。
                         prompt 的 edit_image 数量需与 edit_latents 一致。

        Returns:
            tile_prompt_embs: list[Tensor]，每个元素 shape [1, seq_len, hidden_dim]
            tile_prompt_emb_masks: list[Tensor]，每个元素 shape [1, seq_len]
        """
        _, _, latent_h, latent_w = lq_latents.shape
        tile_px = tile_size * 8

        tile_prompt_embs = []
        tile_prompt_emb_masks = []

        total_tiles = 0
        for h_start in range(0, latent_h - tile_size + 1, tile_stride):
            for w_start in range(0, latent_w - tile_size + 1, tile_stride):
                total_tiles += 1

        print(f"[GeneratorEdit] Encoding per-tile prompts for {total_tiles} tiles...")

        tile_idx = 0
        for h_start in range(0, latent_h - tile_size + 1, tile_stride):
            for w_start in range(0, latent_w - tile_size + 1, tile_stride):
                px_left = w_start * 8
                px_top = h_start * 8
                px_right = px_left + tile_px
                px_bottom = px_top + tile_px

                tile_lq_crop = lq_pil.crop((px_left, px_top, px_right, px_bottom))

                # edit_image 数量和顺序与 edit_latents 一致
                edit_image_parts = [tile_lq_crop]
                if full_lq_pil is not None:
                    edit_image_parts.append(full_lq_pil)
                if ref_pil is not None:
                    edit_image_parts.append(ref_pil)
                edit_image_for_tile = edit_image_parts if len(edit_image_parts) > 1 else edit_image_parts[0]

                tile_result = self.encode_prompt(prompt, edit_image=edit_image_for_tile)
                tile_prompt_embs.append(tile_result["prompt_emb"])
                tile_prompt_emb_masks.append(tile_result["prompt_emb_mask"])
                tile_idx += 1

        print(f"[GeneratorEdit] Per-tile prompt encoding done ({tile_idx} tiles).")
        return tile_prompt_embs, tile_prompt_emb_masks
