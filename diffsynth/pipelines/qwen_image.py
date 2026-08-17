import random
import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from math import prod

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora.merge import merge_lora

from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..models.qwen_image_controlnet import QwenImageBlockWiseControlNet
from ..models.siglip2_image_encoder import Siglip2ImageEncoder
from ..models.dinov3_image_encoder import DINOv3ImageEncoder
from ..models.qwen_image_image2lora import QwenImageImage2LoRAModel


class QwenImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor
        
        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: QwenImageVAE = None
        self.blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None
        self.tokenizer: Qwen2Tokenizer = None
        self.siglip2_image_encoder: Siglip2ImageEncoder = None
        self.dinov3_image_encoder: DINOv3ImageEncoder = None
        self.image2lora_style: QwenImageImage2LoRAModel = None
        self.image2lora_coarse: QwenImageImage2LoRAModel = None
        self.image2lora_fine: QwenImageImage2LoRAModel = None
        self.processor: Qwen2VLProcessor = None
        self.in_iteration_models = ("dit", "blockwise_controlnet")
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_LayerInputImageEmbedder(),
            QwenImageUnit_ContextImageEmbedder(),
            QwenImageUnit_PromptEmbedder(),
            QwenImageUnit_EntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
        ]
        self.model_fn = model_fn_qwen_image
        self.compilable_models = ["dit"]
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(model_pool.fetch_model("qwen_image_blockwise_controlnet", index="all"))
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        pipe.siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
        pipe.dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
        pipe.image2lora_style = model_pool.fetch_model("qwen_image_image2lora_style")
        pipe.image2lora_coarse = model_pool.fetch_model("qwen_image_image2lora_coarse")
        pipe.image2lora_fine = model_pool.fetch_model("qwen_image_image2lora_fine")
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str = "",
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Inpaint
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        exponential_shift_mu: float = None,
        # Blockwise ControlNet
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        # Qwen-Image-Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        # Qwen-Image-Edit-2511
        zero_cond_t: bool = False,
        # Qwen-Image-Layered
        layer_input_image: Image.Image = None,
        layer_num: int = None,
        # In-context control
        context_image: Image.Image = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(height // 16) * (width // 16), exponential_shift_mu=exponential_shift_mu)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_rope_interpolation": edit_rope_interpolation, 
            "context_image": context_image,
            "zero_cond_t": zero_cond_t,
            "layer_input_image": layer_input_image,
            "layer_num": layer_num,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        image = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if layer_num is None:
            image = self.vae_output_to_image(image)
        else:
            image = [self.vae_output_to_image(i, pattern="C H W") for i in image]
        self.load_models_to_device([])

        return image


class QwenImageBlockwiseMultiControlNet(torch.nn.Module):
    def __init__(self, models: list[QwenImageBlockWiseControlNet]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)
        for model in models:
            if hasattr(model, "vram_management_enabled") and getattr(model, "vram_management_enabled"):
                self.vram_management_enabled = True

    def preprocess(self, controlnet_inputs: list[ControlNetInput], conditionings: list[torch.Tensor], **kwargs):
        processed_conditionings = []
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            conditioning = rearrange(conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
            model_output = self.models[controlnet_input.controlnet_id].process_controlnet_conditioning(conditioning)
            processed_conditionings.append(model_output)
        return processed_conditionings

    def blockwise_forward(self, image, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, block_id, **kwargs):
        res = 0
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start + (1e-4) or progress < controlnet_input.end - (1e-4):
                continue
            model_output = self.models[controlnet_input.controlnet_id].blockwise_forward(image, conditioning, block_id)
            res = res + model_output * controlnet_input.scale
        return res


class QwenImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: QwenImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "layer_num"),
            output_params=("noise",),
        )

    def process(self, pipe: QwenImagePipeline, height, width, seed, rand_device, layer_num):
        if layer_num is None:
            noise = pipe.generate_noise((1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        else:
            noise = pipe.generate_noise((layer_num + 1, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class QwenImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        if isinstance(input_image, list):
            input_latents = []
            for image in input_image:
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                input_latents.append(pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
            input_latents = torch.concat(input_latents, dim=0)
        else:
            image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class QwenImageUnit_ConditionImageEmbedder(PipelineUnit):
    """ODTSR 超分条件图像编码 Unit：将 condition_image 编码为 condition_latents 和 condition_rgb。"""
    def __init__(self):
        super().__init__(
            input_params=("condition_image", "tiled", "tile_size", "tile_stride"),
            output_params=("condition_latents", "condition_rgb"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, condition_image, tiled, tile_size, tile_stride):
        if condition_image is None:
            return {}
        pipe.load_models_to_device(['vae'])
        if isinstance(condition_image, list):
            condition_latents_list = []
            condition_rgb_list = []
            for img in condition_image:
                img_rgb = pipe.preprocess_image(img).to(device=pipe.device, dtype=pipe.torch_dtype)
                if tile_size is None:
                    img_latents = pipe.vae.encode(img_rgb, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                else:
                    img_latents = pipe.vae.encode(img_rgb, tiled=tiled, tile_size=tile_size * 8, tile_stride=tile_stride * 8)
                condition_latents_list.append(img_latents)
                condition_rgb_list.append(img_rgb)
            condition_latents = torch.stack(condition_latents_list, dim=0).permute(1, 0, 2, 3, 4)
            condition_rgb = torch.stack(condition_rgb_list, dim=0).permute(1, 0, 2, 3, 4)
        else:
            condition_rgb = pipe.preprocess_image(condition_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            if tile_size is None:
                condition_latents = pipe.vae.encode(condition_rgb, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            else:
                condition_latents = pipe.vae.encode(condition_rgb, tiled=tiled, tile_size=tile_size * 8, tile_stride=tile_stride * 8)
        return {"condition_latents": condition_latents, "condition_rgb": condition_rgb}


class QwenImageUnit_LayerInputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("layer_input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("layer_input_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, layer_input_image, tiled, tile_size, tile_stride):
        if layer_input_image is None:
            return {}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(layer_input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"layer_input_latents": latents}


class QwenImageUnit_Inpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("inpaint_mask", "height", "width", "inpaint_blur_size", "inpaint_blur_sigma"),
            output_params=("inpaint_mask",),
        )

    def process(self, pipe: QwenImagePipeline, inpaint_mask, height, width, inpaint_blur_size, inpaint_blur_sigma):
        if inpaint_mask is None:
            return {}
        inpaint_mask = pipe.preprocess_image(inpaint_mask.convert("RGB").resize((width // 8, height // 8)), min_value=0, max_value=1)
        inpaint_mask = inpaint_mask.mean(dim=1, keepdim=True)
        if inpaint_blur_size is not None and inpaint_blur_sigma is not None:
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=inpaint_blur_size * 2 + 1, sigma=inpaint_blur_sigma)
            inpaint_mask = blur(inpaint_mask)
        return {"inpaint_mask": inpaint_mask}


class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))
    
    def encode_prompt(self, pipe: QwenImagePipeline, prompt):
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.tokenizer(txt, max_length=4096+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
        if model_inputs.input_ids.shape[1] >= 1024:
            print(f"Warning!!! QwenImage model was trained on prompts up to 512 tokens. Current prompt requires {model_inputs['input_ids'].shape[1] - drop_idx} tokens, which may lead to unpredictable behavior.")
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
        
    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
    
    def encode_prompt_edit_multi(self, pipe: QwenImagePipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        base_img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(edit_image))])
        txt = [template.format(base_img_prompt + e) for e in prompt]
        edit_image = [self.resize_image(image) for image in edit_image]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states

    def process(self, pipe: QwenImagePipeline, prompt, edit_image=None) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            prompt = [prompt]
            if edit_image is None:
                split_hidden_states = self.encode_prompt(pipe, prompt)
            elif isinstance(edit_image, Image.Image):
                split_hidden_states = self.encode_prompt_edit(pipe, prompt, edit_image)
            else:
                split_hidden_states = self.encode_prompt_edit_multi(pipe, prompt, edit_image)
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}


class QwenImageUnit_EntityControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("eligen_entity_prompts", "width", "height", "eligen_enable_on_negative", "cfg_scale"),
            output_params=("entity_prompt_emb", "entity_masks", "entity_prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def get_prompt_emb(self, pipe: QwenImagePipeline, prompt) -> dict:
        if pipe.text_encoder is not None:
            prompt = [prompt]
            template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            drop_idx = 34
            txt = [template.format(e) for e in prompt]
            txt_tokens = pipe.tokenizer(txt, max_length=1024+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
            hidden_states = pipe.text_encoder(input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True,)[-1]
            
            split_hidden_states = self.extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}

    def preprocess_masks(self, pipe, masks, height, width, dim):
        out_masks = []
        for mask in masks:
            mask = pipe.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, dim, 1, 1).to(device=pipe.device, dtype=pipe.torch_dtype)
            out_masks.append(mask)
        return out_masks

    def prepare_entity_inputs(self, pipe, entity_prompts, entity_masks, width, height):
        entity_masks = self.preprocess_masks(pipe, entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w
        prompt_embs, prompt_emb_masks = [], []
        for entity_prompt in entity_prompts:
            prompt_emb_dict = self.get_prompt_emb(pipe, entity_prompt)
            prompt_embs.append(prompt_emb_dict['prompt_emb'])
            prompt_emb_masks.append(prompt_emb_dict['prompt_emb_mask'])
        return prompt_embs, prompt_emb_masks, entity_masks

    def prepare_eligen(self, pipe, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, enable_eligen_on_negative, cfg_scale):
        entity_prompt_emb_posi, entity_prompt_emb_posi_mask, entity_masks_posi = self.prepare_entity_inputs(pipe, eligen_entity_prompts, eligen_entity_masks, width, height)
        if enable_eligen_on_negative and cfg_scale != 1.0:
            entity_prompt_emb_nega = [prompt_emb_nega['prompt_emb']] * len(entity_prompt_emb_posi)
            entity_prompt_emb_nega_mask = [prompt_emb_nega['prompt_emb_mask']] * len(entity_prompt_emb_posi)
            entity_masks_nega = entity_masks_posi
        else:
            entity_prompt_emb_nega, entity_prompt_emb_nega_mask, entity_masks_nega = None, None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi, "entity_prompt_emb_mask": entity_prompt_emb_posi_mask}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega, "entity_prompt_emb_mask": entity_prompt_emb_nega_mask}
        return eligen_kwargs_posi, eligen_kwargs_nega

    def process(self, pipe: QwenImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        eligen_entity_prompts, eligen_entity_masks = inputs_shared.get("eligen_entity_prompts", None), inputs_shared.get("eligen_entity_masks", None)
        if eligen_entity_prompts is None or eligen_entity_masks is None or len(eligen_entity_prompts) == 0 or len(eligen_entity_masks) == 0:
            return inputs_shared, inputs_posi, inputs_nega
        pipe.load_models_to_device(self.onload_model_names)
        eligen_enable_on_negative = inputs_shared.get("eligen_enable_on_negative", False)
        eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(pipe, inputs_nega,
            eligen_entity_prompts, eligen_entity_masks, inputs_shared["width"], inputs_shared["height"],
            eligen_enable_on_negative, inputs_shared["cfg_scale"])
        inputs_posi.update(eligen_kwargs_posi)
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update(eligen_kwargs_nega)
        return inputs_shared, inputs_posi, inputs_nega



class QwenImageUnit_BlockwiseControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("blockwise_controlnet_inputs", "tiled", "tile_size", "tile_stride"),
            output_params=("blockwise_controlnet_conditioning",),
            onload_model_names=("vae",)
        )

    def apply_controlnet_mask_on_latents(self, pipe, latents, mask):
        mask = (pipe.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents

    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        mask = mask.resize(image.size)
        mask = pipe.preprocess_image(mask).mean(dim=[0, 1]).cpu()
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image

    def process(self, pipe: QwenImagePipeline, blockwise_controlnet_inputs: list[ControlNetInput], tiled, tile_size, tile_stride):
        if blockwise_controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        conditionings = []
        for controlnet_input in blockwise_controlnet_inputs:
            image = controlnet_input.image
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_image(pipe, image, controlnet_input.inpaint_mask)

            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_latents(pipe, image, controlnet_input.inpaint_mask)
            conditionings.append(image)
            
        return {"blockwise_controlnet_conditioning": conditionings}


class QwenImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "tiled", "tile_size", "tile_stride", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_image"),
            onload_model_names=("vae",)
        )


    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height


    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return edit_image.resize((calculated_width, calculated_height))


    def process(self, pipe: QwenImagePipeline, edit_image, tiled, tile_size, tile_stride, edit_image_auto_resize=False):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            resized_edit_image = self.edit_image_auto_resize(edit_image) if edit_image_auto_resize else edit_image
            edit_image = pipe.preprocess_image(resized_edit_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            edit_latents = pipe.vae.encode(edit_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        else:
            resized_edit_image, edit_latents = [], []
            for image in edit_image:
                if edit_image_auto_resize:
                    image = self.edit_image_auto_resize(image)
                resized_edit_image.append(image)
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                edit_latents.append(latents)
        return {"edit_latents": edit_latents, "edit_image": resized_edit_image}


class QwenImageUnit_Image2LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_images",),
            output_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            onload_model_names=("siglip2_image_encoder", "dinov3_image_encoder", "text_encoder"),
        )
        from ..core.data.operators import ImageCropAndResize
        self.processor_lowres = ImageCropAndResize(height=28*8, width=28*8)
        self.processor_highres = ImageCropAndResize(height=1024, width=1024)

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image):
        prompt = [prompt]
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
        return prompt_embeds.view(1, -1)
    
    def encode_images_using_siglip2(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["siglip2_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.siglip2_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_dinov3(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["dinov3_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.dinov3_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_qwenvl(self, pipe: QwenImagePipeline, images: list[Image.Image], highres=False):
        pipe.load_models_to_device(["text_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image) if highres else self.processor_lowres(image)
            embs.append(self.encode_prompt_edit(pipe, prompt="", edit_image=image))
        embs = torch.stack(embs)
        return embs

    def encode_images(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        if images is None:
            return {}
        if not isinstance(images, list):
            images = [images]
        embs_siglip2 = self.encode_images_using_siglip2(pipe, images)
        embs_dinov3 = self.encode_images_using_dinov3(pipe, images)
        x = torch.concat([embs_siglip2, embs_dinov3], dim=-1)
        residual = None
        residual_highres = None
        if pipe.image2lora_coarse is not None:
            residual = self.encode_images_using_qwenvl(pipe, images, highres=False)
        if pipe.image2lora_fine is not None:
            residual_highres = self.encode_images_using_qwenvl(pipe, images, highres=True)
        return x, residual, residual_highres

    def process(self, pipe: QwenImagePipeline, image2lora_images):
        if image2lora_images is None:
            return {}
        x, residual, residual_highres = self.encode_images(pipe, image2lora_images)
        return {"image2lora_x": x, "image2lora_residual": residual, "image2lora_residual_highres": residual_highres}


class QwenImageUnit_Image2LoRADecode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            output_params=("lora",),
            onload_model_names=("image2lora_coarse", "image2lora_fine", "image2lora_style"),
        )
    
    def process(self, pipe: QwenImagePipeline, image2lora_x, image2lora_residual, image2lora_residual_highres):
        if image2lora_x is None:
            return {}
        loras = []
        if pipe.image2lora_style is not None:
            pipe.load_models_to_device(["image2lora_style"])
            for x in image2lora_x:
                loras.append(pipe.image2lora_style(x=x, residual=None))
        if pipe.image2lora_coarse is not None:
            pipe.load_models_to_device(["image2lora_coarse"])
            for x, residual in zip(image2lora_x, image2lora_residual):
                loras.append(pipe.image2lora_coarse(x=x, residual=residual))
        if pipe.image2lora_fine is not None:
            pipe.load_models_to_device(["image2lora_fine"])
            for x, residual in zip(image2lora_x, image2lora_residual_highres):
                loras.append(pipe.image2lora_fine(x=x, residual=residual))
        lora = merge_lora(loras, alpha=1 / len(image2lora_x))
        return {"lora": lora}


class QwenImageUnit_ContextImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("context_image", "height", "width", "tiled", "tile_size", "tile_stride", "layer_input_image"),
            output_params=("context_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, context_image, height, width, tiled, tile_size, tile_stride, layer_input_image=None):
        if context_image is None:
            return {}
        if layer_input_image is not None:
            context_image = context_image.convert("RGBA")
        pipe.load_models_to_device(self.onload_model_names)
        context_image = pipe.preprocess_image(context_image.resize((width, height))).to(device=pipe.device, dtype=pipe.torch_dtype)
        context_latents = pipe.vae.encode(context_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"context_latents": context_latents}


# ==================== ODTSR SR model_fn 开始 ====================

from ..models.qwen_image_dit import ConditionTypeEmbedding


def _model_fn_qwen_image_sr_core(
    dit, latents, condition_latents, timestep, prompt_emb, prompt_emb_mask,
    height, width, condition_type_embed=None,
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    return_condition_output=False,
):
    """
    ODTSR 超分专用 model_fn 核心逻辑。
    支持 condition_latents (4D 单图 / 5D 多图 / list 任意分辨率) + condition_type_embed。
    """
    img_shapes = [(latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2)]
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000

    image = rearrange(
        latents, "B C (H P) (W Q) -> B (H W) (C P Q)",
        H=height // 16, W=width // 16, P=2, Q=2,
    )
    _, noise_seq_len, _ = image.shape

    if condition_latents is not None:
        if isinstance(condition_latents, (list, tuple)):
            num_conds = len(condition_latents)
            use_high_dim = (
                condition_type_embed is not None
                and condition_type_embed.type_embed.embedding_dim == 3072
            )
            image_controls = []
            for i in range(num_conds):
                cond_i = condition_latents[i]
                h_i, w_i = cond_i.shape[2], cond_i.shape[3]
                ctrl = rearrange(
                    cond_i, "B C (H P) (W Q) -> B (H W) (C P Q)",
                    H=h_i // 2, W=w_i // 2, P=2, Q=2,
                )
                if condition_type_embed is not None and not use_high_dim:
                    ctrl = condition_type_embed(ctrl, i)
                image_controls.append(ctrl)

            image_control = torch.cat(image_controls, dim=1)
            image = torch.cat([image, image_control], dim=1)
            image = dit.img_in(image)

            if use_high_dim:
                offset = noise_seq_len
                for i in range(num_conds):
                    seq_len_i = image_controls[i].shape[1]
                    image[:, offset:offset + seq_len_i, :] = condition_type_embed(
                        image[:, offset:offset + seq_len_i, :], i
                    )
                    offset += seq_len_i

        elif condition_latents.dim() == 5:
            batch_size, num_conds_5d, channels_5d, height_lq, width_lq = condition_latents.shape
            use_high_dim = (
                condition_type_embed is not None
                and condition_type_embed.type_embed.embedding_dim == 3072
            )
            image_controls = []
            for i in range(num_conds_5d):
                ctrl = rearrange(
                    condition_latents[:, i],
                    "B C (H P) (W Q) -> B (H W) (C P Q)",
                    H=height // 16, W=width // 16, P=2, Q=2,
                )
                if condition_type_embed is not None and not use_high_dim:
                    ctrl = condition_type_embed(ctrl, i)
                image_controls.append(ctrl)

            image_control = torch.cat(image_controls, dim=1)
            image = torch.cat([image, image_control], dim=1)
            image = dit.img_in(image)

            if use_high_dim:
                seq_per_cond = image_controls[0].shape[1]
                for i in range(num_conds_5d):
                    start = noise_seq_len + i * seq_per_cond
                    end = start + seq_per_cond
                    image[:, start:end, :] = condition_type_embed(
                        image[:, start:end, :], i
                    )

        elif condition_latents.dim() == 4:
            image_control = rearrange(
                condition_latents, "B C (H P) (W Q) -> B (H W) (C P Q)",
                H=height // 16, W=width // 16, P=2, Q=2,
            )
            image = torch.cat([image, image_control], dim=1)
            image = dit.img_in(image)
        else:
            raise ValueError("condition_latents.dim() must be 4 or 5")
    else:
        image = dit.img_in(image)

    text = dit.txt_in(dit.txt_norm(prompt_emb))
    conditioning = dit.time_text_embed(timestep, image.dtype)
    image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

    # 为条件图像构造独立的 RoPE
    if condition_latents is not None:
        if isinstance(condition_latents, (list, tuple)):
            control_rope_parts = []
            for i in range(len(condition_latents)):
                h_i, w_i = condition_latents[i].shape[2], condition_latents[i].shape[3]
                ctrl_img_shapes = [(condition_latents[i].shape[0], h_i // 2, w_i // 2)]
                ctrl_rope = dit.pos_embed(ctrl_img_shapes, txt_seq_lens, device=latents.device)
                control_rope_parts.append(ctrl_rope[0])
            image_rotary_emb_control = (
                torch.cat([image_rotary_emb[0]] + control_rope_parts, dim=0),
                image_rotary_emb[1],
            )
        elif condition_latents.dim() == 5:
            num_conds_rope = condition_latents.shape[1]
            image_rotary_emb_control = (
                torch.cat([image_rotary_emb[0]] * (1 + num_conds_rope), dim=0),
                image_rotary_emb[1],
            )
        else:
            image_rotary_emb_control = (
                torch.cat([image_rotary_emb[0], image_rotary_emb[0]], dim=0),
                image_rotary_emb[1],
            )
    else:
        image_rotary_emb_control = image_rotary_emb

    for block in dit.transformer_blocks:
        text, image = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            image=image,
            text=text,
            temb=conditioning,
            image_rotary_emb=image_rotary_emb_control,
        )

    # 从拼接序列中取出噪声部分
    if condition_latents is not None:
        if return_condition_output:
            condition_tokens = image[:, noise_seq_len:, :]
            condition_tokens = dit.norm_out(condition_tokens, conditioning)
            condition_tokens = dit.proj_out(condition_tokens)
        image = image[:, :noise_seq_len, :]

    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)

    result = rearrange(
        image, "B (H W) (C P Q) -> B C (H P) (W Q)",
        H=height // 16, W=width // 16, P=2, Q=2,
    )
    return result


def tiled_model_fn_qwen_image_sr(
    dit, latents, condition_latents, timestep, prompt_emb, prompt_emb_mask,
    tile_size, tile_stride, condition_type_embed=None,
    global_ref_latents=None, global_lq_latents=None,
    tile_prompt_embs=None, tile_prompt_emb_masks=None,
):
    """分块推理 model_fn：逐块调用 _model_fn_qwen_image_sr_core 并加权合并。"""
    _, _, latent_h, latent_w = latents.shape

    output = torch.zeros_like(latents)
    count = torch.zeros(1, 1, latent_h, latent_w, device=latents.device, dtype=latents.dtype)

    tile_idx = 0
    for h_start in range(0, latent_h - tile_size + 1, tile_stride):
        for w_start in range(0, latent_w - tile_size + 1, tile_stride):
            h_end = h_start + tile_size
            w_end = w_start + tile_size

            tile_latent = latents[:, :, h_start:h_end, w_start:w_end]

            if condition_latents is not None:
                if isinstance(condition_latents, (list, tuple)):
                    tile_cond = [
                        c[:, :, h_start:h_end, w_start:w_end]
                        if c.shape[2:] == latents.shape[2:] else c
                        for c in condition_latents
                    ]
                elif condition_latents.dim() == 5:
                    tile_cond = condition_latents[:, :, :, h_start:h_end, w_start:w_end]
                elif condition_latents.dim() == 4:
                    tile_cond = condition_latents[:, :, h_start:h_end, w_start:w_end]
                else:
                    tile_cond = condition_latents
            else:
                tile_cond = None

            if tile_prompt_embs is not None and tile_idx < len(tile_prompt_embs):
                tile_pe = tile_prompt_embs[tile_idx]
                tile_pm = tile_prompt_emb_masks[tile_idx]
            else:
                tile_pe = prompt_emb
                tile_pm = prompt_emb_mask

            if global_ref_latents is not None or global_lq_latents is not None:
                parts = []
                if tile_cond is not None:
                    parts = list(tile_cond) if isinstance(tile_cond, (list, tuple)) else [tile_cond]
                if global_lq_latents is not None:
                    parts.append(global_lq_latents)
                if global_ref_latents is not None:
                    parts.append(global_ref_latents)
                tile_cond = parts if len(parts) > 1 else parts[0]

            tile_out = _model_fn_qwen_image_sr_core(
                dit=dit, latents=tile_latent, condition_latents=tile_cond,
                timestep=timestep, prompt_emb=tile_pe, prompt_emb_mask=tile_pm,
                height=tile_size * 8, width=tile_size * 8,
                condition_type_embed=condition_type_embed,
            )

            output[:, :, h_start:h_end, w_start:w_end] += tile_out
            count[:, :, h_start:h_end, w_start:w_end] += 1
            tile_idx += 1

    return output / count


def model_fn_qwen_image_sr(
    dit=None, latents=None, condition_latents=None, timestep=None,
    prompt_emb=None, prompt_emb_mask=None, height=None, width=None,
    use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False,
    condition_type_embed=None, **kwargs
):
    """ODTSR 超分推理入口。自动判断是否使用 tiled 推理。"""
    if kwargs.get('tiled', False):
        return tiled_model_fn_qwen_image_sr(
            dit, latents, condition_latents, timestep, prompt_emb, prompt_emb_mask,
            kwargs.get('tile_size'), kwargs.get('tile_stride'),
            condition_type_embed,
            kwargs.get('global_ref_latents'),
            kwargs.get('global_lq_latents'),
            kwargs.get('tile_prompt_embs'),
            kwargs.get('tile_prompt_emb_masks'),
        )
    return _model_fn_qwen_image_sr_core(
        dit, latents, condition_latents, timestep, prompt_emb, prompt_emb_mask,
        height, width, condition_type_embed,
        use_gradient_checkpointing, use_gradient_checkpointing_offload,
        kwargs.get('return_condition_output', False),
    )

# ==================== ODTSR SR model_fn 结束 ====================


def model_fn_qwen_image(
    dit: QwenImageDiT = None,
    blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    blockwise_controlnet_conditioning=None,
    blockwise_controlnet_inputs=None,
    progress_id=0,
    num_inference_steps=1,
    entity_prompt_emb=None,
    entity_prompt_emb_mask=None,
    entity_masks=None,
    edit_latents=None,
    layer_input_latents=None,
    layer_num=None,
    context_latents=None,
    enable_fp8_attention=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    edit_rope_interpolation=False,
    zero_cond_t=False,
    lq_mask=None,
    ref_mask=None,
    **kwargs
):
    if layer_num is None:
        layer_num = 1
        img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)]
    else:
        layer_num = layer_num + 1
        img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)] * layer_num
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000
    
    image = rearrange(latents, "(B N) C (H P) (W Q) -> B (N H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2, N=layer_num)
    image_seq_len = image.shape[1]

    if context_latents is not None:
        img_shapes += [(context_latents.shape[0], context_latents.shape[2]//2, context_latents.shape[3]//2)]
        context_image = rearrange(context_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=context_latents.shape[2]//2, W=context_latents.shape[3]//2, P=2, Q=2)
        image = torch.cat([image, context_image], dim=1)
    if edit_latents is not None:
        edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
        img_shapes += [(e.shape[0], e.shape[2]//2, e.shape[3]//2) for e in edit_latents_list]
        edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2) for e in edit_latents_list]
        image = torch.cat([image] + edit_image, dim=1)
    if layer_input_latents is not None:
        layer_num = layer_num + 1
        img_shapes += [(layer_input_latents.shape[0], layer_input_latents.shape[2]//2, layer_input_latents.shape[3]//2)]
        layer_input_latents = rearrange(layer_input_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        image = torch.cat([image, layer_input_latents], dim=1)

    image = dit.img_in(image)
    if zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in [img_shapes]],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None
    conditioning = dit.time_text_embed(
        timestep,
        image.dtype,
        addition_t_cond=None if not dit.time_text_embed.use_additional_t_cond else torch.tensor([0]).to(device=image.device, dtype=torch.long)
    )

    if entity_prompt_emb is not None:
        text, image_rotary_emb, attention_mask = dit.process_entity_masks(
            latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask,
            entity_masks, height, width, image, img_shapes,
        )
    else:
        text = dit.txt_in(dit.txt_norm(prompt_emb))
        if edit_rope_interpolation:
            image_rotary_emb = dit.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=latents.device)
        else:
            image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        attention_mask = None

    # control attention mask
    use_attention_mask = False
    if lq_mask is not None and ref_mask is not None and use_attention_mask:
        # soft patchify（保留浮点连续值）
        lq_mask_patch = rearrange(lq_mask.float(), "B (H P) (W Q) -> B (H W) (P Q)", P=2, Q=2).mean(dim=-1)  # [B, N_lq]
        ref_mask_patch = rearrange(ref_mask.float(), "B (H P) (W Q) -> B (H W) (P Q)", P=2, Q=2).mean(dim=-1)  # [B, N_ref]

        S_noise = math.prod(img_shapes[0])
        S_lq    = math.prod(img_shapes[1])
        S_full  = math.prod(img_shapes[2]) if len(img_shapes) > 3 else 0  # full_lq 可能不存在
        S_ref   = math.prod(img_shapes[-1])
        S_txt   = text.shape[1]
        S_img   = image.shape[1]
        assert lq_mask_patch.shape[1] == math.prod(image.shape[0]), \
            f"lq_mask_patch seq_len {lq_mask_patch.shape[1]} != noise img_shape {math.prod(img_shapes[1])}"
        assert lq_mask_patch.shape[1] == math.prod(image.shape[1]), \
            f"lq_mask_patch seq_len {lq_mask_patch.shape[1]} != lq img_shape {math.prod(img_shapes[1])}"
        assert ref_mask_patch.shape[1] == math.prod(image.shape[-1]), \
            f"ref_mask_patch seq_len {ref_mask_patch.shape[1]} != ref img_shape {math.prod(img_shapes[-1])}"

        total = S_txt + S_img
        # additive bias，dtype 用浮点
        attention_mask = torch.zeros((image.shape[0], 1, total, total), device=image.device, dtype=image.dtype)
        
        # 训练时 alpha 随机范围
        alpha_lq = random.uniform(0.5, 3.0)
        alpha_ref = random.uniform(0.5, 3.0)
        # noise → lq: LQ mask 越低越需要 lq（自身信息充足）
        noise_start = S_txt
        noise_end   = S_txt + S_noise
        lq_start    = noise_end
        lq_end      = lq_start + S_lq
        attention_mask[:, :, noise_start:noise_end, lq_start:lq_end] = \
            alpha_lq * (1 - lq_mask_patch).unsqueeze(1).unsqueeze(-1)  # [B, 1, S_noise, S_lq]

        # noise → ref: 需要ref × ref有效
        ref_start = S_txt + S_img - S_ref  # ref 在 image 末尾
        ref_end   = S_txt + S_img
        attention_mask[:, :, noise_start:noise_end, ref_start:ref_end] = \
            alpha_ref * (lq_mask_patch.unsqueeze(1).unsqueeze(-1) * ref_mask_patch.unsqueeze(1).unsqueeze(2))  # [B, 1, S_noise, S_ref]

    if blockwise_controlnet_conditioning is not None:
        blockwise_controlnet_conditioning = blockwise_controlnet.preprocess(
            blockwise_controlnet_inputs, blockwise_controlnet_conditioning)

    for block_id, block in enumerate(dit.transformer_blocks):
        text, image = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            image=image,
            text=text,
            temb=conditioning,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
            modulate_index=modulate_index,
        )
        if blockwise_controlnet_conditioning is not None:
            image_slice = image[:, :image_seq_len].clone()
            controlnet_output = blockwise_controlnet.blockwise_forward(
                image=image_slice, conditionings=blockwise_controlnet_conditioning,
                controlnet_inputs=blockwise_controlnet_inputs, block_id=block_id,
                progress_id=progress_id, num_inference_steps=num_inference_steps,
            )
            image[:, :image_seq_len] = image_slice + controlnet_output
    
    if zero_cond_t:
        conditioning = conditioning.chunk(2, dim=0)[0]
    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)
    image = image[:, :image_seq_len]
    
    latents = rearrange(image, "B (N H W) (C P Q) -> (B N) C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2, B=1)
    return latents


def tiled_model_fn_qwen_image(
    dit=None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    tile_size=64,
    tile_stride=48,
    edit_latents=None,
    zero_cond_t=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    tile_prompt_embs=None,
    tile_prompt_emb_masks=None,
    all_edit_latents_global=False,
    **kwargs,
):
    """
    Edit 模式的分块推理：
    - 噪声 latent 按 tile 切分
    - LQ edit_latent 同步按 tile 切分（与 noise 同尺寸，保持训练一致性）
    - REF edit_latent 不切（全局可见，提供全局参考信息）

    Args:
        tile_size: latent 空间的 tile 大小（像素空间为 tile_size * 8）
        tile_stride: latent 空间的 tile 步长（< tile_size 则有 overlap）
        tile_prompt_embs: 可选，per-tile 的 prompt embeddings 列表（行优先顺序），
                          长度应等于 tile 总数。若为 None 则所有 tile 共享全局 prompt_emb。
        tile_prompt_emb_masks: 可选，与 tile_prompt_embs 配对的 attention masks 列表。
        all_edit_latents_global: 若为 True，所有 edit_latents 均为全局条件，不按 tile 切分
                                 （用于 drop_lq_crop_condition 场景，edit_latents 只含 full_lq + ref）。
    """
    _, _, latent_h, latent_w = latents.shape

    output = torch.zeros_like(latents)
    count = torch.zeros(1, 1, latent_h, latent_w, device=latents.device, dtype=latents.dtype)

    # 构建 overlap mask，重叠区域线性权重融合
    border_width = tile_size - tile_stride
    if border_width > 0:
        mask_x = torch.arange(tile_size).repeat(tile_size, 1).T
        mask_y = torch.arange(tile_size).repeat(tile_size, 1)
        mask = torch.stack([mask_x + 1, tile_size - mask_x, mask_y + 1, tile_size - mask_y]).min(dim=0).values
        mask = (mask / border_width).clip(0, 1)
        mask = mask.to(device=latents.device, dtype=latents.dtype)
        mask = rearrange(mask, "h w -> 1 1 h w")
    else:
        mask = torch.ones(1, 1, tile_size, tile_size, device=latents.device, dtype=latents.dtype)

    # 分离 edit_latents：
    # all_edit_latents_global=True 时所有条件都全局不切（drop_lq_crop_condition 场景）
    # 否则第一个是 LQ（按 tile 切分），其余保持全局
    if all_edit_latents_global:
        lq_edit_latent = None
        global_edit_latents = list(edit_latents) if isinstance(edit_latents, (list, tuple)) else [edit_latents]
    elif isinstance(edit_latents, (list, tuple)):
        lq_edit_latent = edit_latents[0]
        global_edit_latents = list(edit_latents[1:])  # 完整 LQ / REF 等，全局不切分
    else:
        lq_edit_latent = edit_latents
        global_edit_latents = []

    tile_idx = 0
    for h_start in range(0, latent_h - tile_size + 1, tile_stride):
        for w_start in range(0, latent_w - tile_size + 1, tile_stride):
            h_end = h_start + tile_size
            w_end = w_start + tile_size

            tile_latent = latents[:, :, h_start:h_end, w_start:w_end]

            # 构建当前 tile 的 edit_latents
            if all_edit_latents_global:
                tile_edit_latents = global_edit_latents if len(global_edit_latents) > 1 else global_edit_latents[0]
            else:
                # LQ edit_latent 同步切分（与 noise tile 同尺寸，保持训练一致性）
                tile_lq_edit = lq_edit_latent[:, :, h_start:h_end, w_start:w_end]
                if global_edit_latents:
                    tile_edit_latents = [tile_lq_edit] + global_edit_latents
                else:
                    tile_edit_latents = tile_lq_edit

            # 选择当前 tile 的 prompt embedding（per-tile 或全局共享）
            if tile_prompt_embs is not None:
                current_prompt_emb = tile_prompt_embs[tile_idx]
                current_prompt_emb_mask = tile_prompt_emb_masks[tile_idx]
            else:
                current_prompt_emb = prompt_emb
                current_prompt_emb_mask = prompt_emb_mask

            tile_out = model_fn_qwen_image(
                dit=dit,
                latents=tile_latent,
                timestep=timestep,
                prompt_emb=current_prompt_emb,
                prompt_emb_mask=current_prompt_emb_mask,
                height=tile_size * 8,
                width=tile_size * 8,
                edit_latents=tile_edit_latents,
                zero_cond_t=zero_cond_t,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                **kwargs,
            )

            output[:, :, h_start:h_end, w_start:w_end] += tile_out * mask
            count[:, :, h_start:h_end, w_start:w_end] += mask
            tile_idx += 1

    return output / count
