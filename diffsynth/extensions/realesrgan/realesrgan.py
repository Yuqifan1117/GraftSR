import sys
import torchvision

# 检查 torchvision 版本，如果 >= 0.15，则应用补丁
if tuple(map(int, torchvision.__version__.split('.')[:2])) >= (0, 15):
    try:
        # 尝试导入新路径
        from torchvision.transforms import _functional_tensor as new_ft
        # 创建一个假的旧模块对象
        import types
        old_ft = types.ModuleType('torchvision.transforms.functional_tensor')
        # 将新模块的内容复制到假模块中
        for attr in dir(new_ft):
            if not attr.startswith('__'):
                setattr(old_ft, attr, getattr(new_ft, attr))
        # 将假模块注册到 sys.modules，伪装成旧模块
        sys.modules['torchvision.transforms.functional_tensor'] = old_ft
        print("Applied patch for torchvision >= 0.15 compatibility.")
    except ImportError:
        pass

import os
import subprocess
import tempfile
import numpy as np
import cv2
import glob
import math
import yaml
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F

from basicsr.data.transforms import augment
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize, rgb_to_grayscale)

# cur_path = os.path.dirname(os.path.abspath(__file__))

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)  # ignore_security_alert_wait_for_fix RCE

    return opt

class RealESRGAN_degradation(object):
    def __init__(self, opt_path='params_realesrgan.yml', device='cpu'):
        # opt_path = f'{cur_path}/{opt_name}'
        self.opt = opt_parse(opt_path)
        self.device = device #torch.device('cpu')
        optk = self.opt['kernel_info']       

        # blur settings for the first degradation
        self.blur_kernel_size = optk['blur_kernel_size']
        self.kernel_list = optk['kernel_list']
        self.kernel_prob = optk['kernel_prob']
        self.blur_sigma = optk['blur_sigma']
        self.betag_range = optk['betag_range']
        self.betap_range = optk['betap_range']
        self.sinc_prob = optk['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = optk['blur_kernel_size2']
        self.kernel_list2 = optk['kernel_list2']
        self.kernel_prob2 = optk['kernel_prob2']
        self.blur_sigma2 = optk['blur_sigma2']
        self.betag_range2 = optk['betag_range2']
        self.betap_range2 = optk['betap_range2']
        self.sinc_prob2 = optk['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = optk['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)
        self.usm_shaper = USMSharp().to(self.device)
    
    def color_jitter_pt(self, img, brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def random_augment(self, img_gt):
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=False, rotation=False, return_status=True)
        """
        # random color jitter 
        if np.random.uniform() < self.opt['color_jitter_prob']:
            jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
            img_gt = img_gt + jitter_val
            img_gt = np.clip(img_gt, 0, 1)    

        # random grayscale
        if np.random.uniform() < self.opt['gray_prob']:
            #img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2GRAY)
            img_gt = np.tile(img_gt[:, :, None], [1, 1, 3])
        """
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=False, float32=True)[0].unsqueeze(0)

        return img_gt

    def random_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2) 

        return kernel, kernel2, sinc_kernel

    @staticmethod
    def _kernel_blur_score(kernel_tensor):
        """
        从模糊核估计归一化的模糊强度 (0~1)。
        使用 kernel 的加权均方根半径（有效 sigma 的代理）来衡量模糊程度。
        pulse kernel（无模糊）得分接近 0，宽高斯核得分接近 1。
        """
        k = kernel_tensor.cpu().float().numpy()
        center = k.shape[0] // 2
        y, x = np.mgrid[:k.shape[0], :k.shape[1]]
        y = y.astype(np.float64) - center
        x = x.astype(np.float64) - center
        radius_squared = x ** 2 + y ** 2
        effective_radius = np.sqrt(np.sum(k * radius_squared))
        max_radius = float(center)
        return min(effective_radius / max(max_radius, 1.0), 1.0)

    @torch.no_grad()
    def degrade_process(self, img_gt, resize_bak=False):
        img_gt = self.random_augment(img_gt)
        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        #img_gt = self.usm_shaper(img_gt) # shaper gt
        ori_h, ori_w = img_gt.size()[2:4]

        scale_final = self.opt['scale']
        if isinstance(scale_final, list):
            scale_final = random.randint(scale_final[0], scale_final[1])

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(img_gt, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale_final * scale), int(ori_w / scale_final * scale)), mode=mode)
        # noise
        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        if np.random.uniform() < self.opt['gray_prob']:
            out = rgb_to_grayscale(out, num_output_channels=1)

        if np.random.uniform() < self.opt['color_jitter_prob']:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
        # clamp and round
        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lq

    # ======================== 带退化分数的退化（用于双退化训练） ======================== #

    @torch.no_grad()
    def degrade_process_with_score(self, img_gt, resize_bak=False):
        """
        与 degrade_process 相同的退化管线，但额外返回一个退化分数。
        分数基于实际采样到的退化参数计算，越大表示退化越重。

        相比原版改进点：
          1. 新增 blur1 分数：第一阶段模糊是必然发生的，用 kernel 有效半径量化其强度
          2. blur2 改为连续值：用 kernel 有效半径而非二值，区分轻/重模糊
          3. 新增 sinc 分数：最终 sinc filter 的振铃效应（脉冲核=无退化，宽核=强振铃）
          4. 新增 gray / color_jitter 分数：灰度化和色彩偏移会显著改变图像内容
          5. 调整权重分配：resize1 降权（可能上采样不产生退化），各分量更均衡

        Returns:
            (img_gt, img_lq, degrade_score): GT tensor, LQ tensor, 退化分数 (float, 0~1)
        """
        img_gt = self.random_augment(img_gt)
        kernel1, kernel2, sinc_kernel = self.random_kernels()
        img_gt, kernel1, kernel2, sinc_kernel = img_gt.to(self.device), kernel1.to(self.device), kernel2.to(self.device), sinc_kernel.to(self.device)
        ori_h, ori_w = img_gt.size()[2:4]

        score_components = {}

        scale_final = self.opt['scale']
        if isinstance(scale_final, list):
            scale_final = random.randint(scale_final[0], scale_final[1])
        max_scale = self.opt['scale'][1] if isinstance(self.opt['scale'], list) else self.opt['scale']
        score_components['scale_final'] = (scale_final - 1) / max(max_scale - 1, 1)

        # ---- 第一阶段 ----
        out = filter2D(img_gt, kernel1)
        score_components['blur1'] = self._kernel_blur_score(kernel1)

        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, self.opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(self.opt['resize_range'][0], 1)
        else:
            scale = 1
        score_components['resize1'] = max(0, 1 - scale)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        gray_noise_prob = self.opt['gray_noise_prob']
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            noise_sigma = np.random.uniform(*self.opt['noise_range'])
            score_components['noise1'] = noise_sigma / max(self.opt['noise_range'][1], 1)
            out = random_add_gaussian_noise_pt(
                out, sigma_range=[noise_sigma, noise_sigma], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            poisson_scale = np.random.uniform(*self.opt['poisson_scale_range'])
            score_components['noise1'] = poisson_scale / max(self.opt['poisson_scale_range'][1], 1)
            out = random_add_poisson_noise_pt(
                out, scale_range=[poisson_scale, poisson_scale],
                gray_prob=gray_noise_prob, clip=True, rounds=False)

        jpeg_q1 = np.random.uniform(*self.opt['jpeg_range'])
        score_components['jpeg1'] = 1 - jpeg_q1 / 100.0
        jpeg_p = out.new_zeros(out.size(0)).fill_(jpeg_q1)
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ---- 第二阶段 ----
        use_second_blur = np.random.uniform() < self.opt['second_blur_prob']
        if use_second_blur:
            out = filter2D(out, kernel2)
            score_components['blur2'] = self._kernel_blur_score(kernel2)
        else:
            score_components['blur2'] = 0.0

        updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale2 = np.random.uniform(1, self.opt['resize_range2'][1])
        elif updown_type == 'down':
            scale2 = np.random.uniform(self.opt['resize_range2'][0], 1)
        else:
            scale2 = 1
        score_components['resize2'] = max(0, 1 - scale2)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / scale_final * scale2), int(ori_w / scale_final * scale2)), mode=mode)

        gray_noise_prob = self.opt['gray_noise_prob2']
        if np.random.uniform() < self.opt['gaussian_noise_prob2']:
            noise_sigma2 = np.random.uniform(*self.opt['noise_range2'])
            score_components['noise2'] = noise_sigma2 / max(self.opt['noise_range2'][1], 1)
            out = random_add_gaussian_noise_pt(
                out, sigma_range=[noise_sigma2, noise_sigma2], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            poisson_scale2 = np.random.uniform(*self.opt['poisson_scale_range2'])
            score_components['noise2'] = poisson_scale2 / max(self.opt['poisson_scale_range2'][1], 1)
            out = random_add_poisson_noise_pt(
                out, scale_range=[poisson_scale2, poisson_scale2],
                gray_prob=gray_noise_prob, clip=True, rounds=False)

        jpeg_q2 = np.random.uniform(*self.opt['jpeg_range2'])
        score_components['jpeg2'] = 1 - jpeg_q2 / 100.0

        # sinc filter 分数：pulse_tensor 得分 ≈ 0（无振铃），宽 sinc 核得分高
        score_components['sinc'] = self._kernel_blur_score(sinc_kernel)

        if np.random.uniform() < 0.5:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)
            jpeg_p = out.new_zeros(out.size(0)).fill_(jpeg_q2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            jpeg_p = out.new_zeros(out.size(0)).fill_(jpeg_q2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // scale_final, ori_w // scale_final), mode=mode)
            out = filter2D(out, sinc_kernel)

        use_gray = np.random.uniform() < self.opt['gray_prob']
        score_components['gray'] = 1.0 if use_gray else 0.0
        if use_gray:
            out = rgb_to_grayscale(out, num_output_channels=1)

        use_color_jitter = np.random.uniform() < self.opt['color_jitter_prob']
        if use_color_jitter:
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)
            score_components['color_jitter'] = 1.0
        else:
            score_components['color_jitter'] = 0.0

        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        # ---- 计算综合退化分数 ----
        # 权重设计原则：
        #   - scale_final 权重最高：直接决定信息损失量，是超分任务的核心退化因子
        #   - blur1 > blur2：第一阶段模糊必然发生且影响被后续退化放大（级联效应）
        #   - resize1 适中：可能上采样（score=0）或下采样，不如 blur1 稳定
        #   - noise1 > noise2, jpeg1 > jpeg2：第一阶段退化的影响被第二阶段放大
        #   - sinc：振铃效应对感知质量影响显著但概率较低
        #   - gray / color_jitter：二值事件，发生时影响大但对"退化轻重"排序贡献有限
        weights = {
            'scale_final': 2.0,
            'blur1': 1.5,
            'resize1': 1.5,
            'noise1': 1.5,
            'jpeg1': 1.2,
            'blur2': 1.0,
            'resize2': 1.0,
            'noise2': 1.0,
            'jpeg2': 0.8,
            'sinc': 0.5,
            'gray': 0.3,
            'color_jitter': 0.2,
        }
        total_weight = sum(weights.values())
        degrade_score = sum(weights[k] * score_components.get(k, 0) for k in weights) / total_weight

        return img_gt, img_lq, degrade_score

    # ======================== 轻量退化（用于参考图） ======================== #

    @torch.no_grad()
    def light_degrade_process(self, img_ref, resize_bak=True):
        """
        轻量退化流程，用于对参考图施加轻微退化，缩小训练时（干净 ref）与推理时（用户提供的非完美 ref）之间的 domain gap。

        退化链路（均由概率独立控制，可叠加）：
          1. 轻微高斯模糊（模拟轻微失焦 / 压缩模糊）
          2. 轻微高斯噪声（模拟传感器噪声）
          3. JPEG 压缩（模拟社交媒体 / 保存压缩）
          4. resize 抖动（先缩小再放大，模拟分辨率损失）

        Args:
            img_ref: 输入参考图 (numpy, [H,W,3], float32, 0-1)
            resize_bak: 是否 resize 回原始尺寸（默认 True，保持 ref 尺寸不变）

        Returns:
            (img_ref, img_ref_lq): 原图 tensor [1,C,H,W] 和退化后的 tensor [1,C,H,W]
        """
        # numpy -> tensor
        img_ref = self.random_augment(img_ref)
        img_ref = img_ref.to(self.device)
        ori_h, ori_w = img_ref.size()[2:4]

        lopt = self.opt.get('light_degrade', {})

        out = img_ref.clone()

        # ==================== Step 1: 模糊（复用 degrade_process 的 random_mixed_kernels）==================== #
        if np.random.uniform() < lopt.get('blur_prob', 0.4):
            kernel_size_range = lopt.get('blur_kernel_range', [3, 7])
            kernel_size = random.choice(
                range(kernel_size_range[0], kernel_size_range[1] + 1, 2)
            )

            light_sinc_prob = lopt.get('sinc_prob', 0.0)
            if np.random.uniform() < light_sinc_prob:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                light_kernel_list = lopt.get('kernel_list', self.kernel_list)
                light_kernel_prob = lopt.get('kernel_prob', self.kernel_prob)
                light_blur_sigma = lopt.get('blur_sigma_range', [0.2, 1.0])
                light_betag_range = lopt.get('betag_range', self.betag_range)
                light_betap_range = lopt.get('betap_range', self.betap_range)
                kernel = random_mixed_kernels(
                    light_kernel_list,
                    light_kernel_prob,
                    kernel_size,
                    light_blur_sigma,
                    light_blur_sigma,
                    [-math.pi, math.pi],
                    light_betag_range,
                    light_betap_range,
                    noise_range=None,
                )

            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
            kernel = torch.FloatTensor(kernel).to(self.device)
            out = filter2D(out, kernel)

        # ==================== Step 2: 轻微高斯噪声 ==================== #
        if np.random.uniform() < lopt.get('noise_prob', 0.3):
            noise_sigma_range = lopt.get('noise_sigma_range', [1, 8])
            out = random_add_gaussian_noise_pt(
                out, sigma_range=noise_sigma_range,
                clip=True, rounds=False, gray_prob=0.0,
            )

        # ==================== Step 3: JPEG 压缩 ==================== #
        if np.random.uniform() < lopt.get('jpeg_prob', 0.5):
            jpeg_range = lopt.get('jpeg_range', [60, 95])
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        # ==================== Step 4: resize 抖动 ==================== #
        if np.random.uniform() < lopt.get('resize_prob', 0.3):
            resize_range = lopt.get('resize_range', [0.6, 0.9])
            scale = np.random.uniform(*resize_range)
            small_h = max(16, int(ori_h * scale))
            small_w = max(16, int(ori_w * scale))
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(small_h, small_w), mode=mode)
            mode = random.choice(['bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

        # ==================== resize back ==================== #
        if resize_bak:
            current_h, current_w = out.size()[2:4]
            if current_h != ori_h or current_w != ori_w:
                mode = random.choice(['bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

        img_ref_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_ref, img_ref_lq

    @torch.no_grad()
    def dual_degrade_process(self, img_gt1, img_gt2, resize_bak=False):
        """
        对两张图用完全相同的退化参数做退化。

        用途：退化方向训练中，对 GT 和 REF HQ 施加一致的退化，使得 LQ 和 REF LQ 的退化风格完全匹配。

        实现方式：先采样一次 kernels，然后用固定 seed 对两张图分别执行 degrade_process，保证所有中间随机量一致。

        注意：此方法不影响 degrade_process 的任何行为——degrade_process 本身完全不变，现有训练不受影响。

        Args:
            img_gt1: 第一张输入图（numpy, [H,W,3], 0-1）
            img_gt2: 第二张输入图（numpy, [H,W,3], 0-1）
            resize_bak: 是否 resize 回各自的原始尺寸

        Returns:
            (gt1, lq1, gt2, lq2): 四个 tensor，lq1 和 lq2 退化风格完全一致
        """
        import random as py_random

        # 采样一个共享 seed
        shared_seed = py_random.randint(0, 2**32 - 1)

        # 先采样一组 kernels（只采一次，确保两张图用同一组模糊核）
        kernels = self.random_kernels()

        # 临时替换 random_kernels，让 degrade_process 复用预采样的 kernels
        original_random_kernels = self.random_kernels
        self.random_kernels = lambda: kernels

        try:
            # 保存当前随机状态
            py_state = py_random.getstate()
            np_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()

            # 对第一张图退化
            py_random.seed(shared_seed)
            np.random.seed(shared_seed % (2**31))
            torch.manual_seed(shared_seed)
            gt1, lq1 = self.degrade_process(img_gt1, resize_bak=resize_bak)

            # 对第二张图用完全相同的随机状态退化
            py_random.seed(shared_seed)
            np.random.seed(shared_seed % (2**31))
            torch.manual_seed(shared_seed)
            gt2, lq2 = self.degrade_process(img_gt2, resize_bak=resize_bak)

            # 恢复原始随机状态，避免影响后续采样
            py_random.setstate(py_state)
            np.random.set_state(np_state)
            torch.random.set_rng_state(torch_state)
        finally:
            # 恢复原始 random_kernels 方法
            self.random_kernels = original_random_kernels

        return gt1, lq1, gt2, lq2

    # ======================== 视频截图退化 ======================== #

    def _yuv420_chroma_subsample(self, img_tensor):
        """
        纯 tensor 操作模拟 YUV420 色度下采样。
        视频标准 4:2:0：亮度全分辨率，色度水平和垂直各减半。
        """
        img = img_tensor.squeeze(0)  # [C, H, W]
        r, g, b = img[0:1], img[1:2], img[2:3]

        # RGB -> YCbCr (BT.601)
        y  =  0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
        cr =  0.5 * r - 0.418688 * g - 0.081312 * b + 0.5

        # 对 CbCr 做 2x 下采样再上采样
        cb_down = F.avg_pool2d(cb.unsqueeze(0), kernel_size=2, stride=2)
        cr_down = F.avg_pool2d(cr.unsqueeze(0), kernel_size=2, stride=2)
        cb_up = F.interpolate(
            cb_down, size=img.shape[1:], mode='bilinear', align_corners=False
        ).squeeze(0)
        cr_up = F.interpolate(
            cr_down, size=img.shape[1:], mode='bilinear', align_corners=False
        ).squeeze(0)

        # YCbCr -> RGB
        cb_s = cb_up - 0.5
        cr_s = cr_up - 0.5
        r_out = y + 1.402 * cr_s
        g_out = y - 0.344136 * cb_s - 0.714136 * cr_s
        b_out = y + 1.772 * cb_s

        return torch.clamp(
            torch.cat([r_out, g_out, b_out], dim=0).unsqueeze(0), 0, 1
        )

    def _motion_blur(self, img_tensor, kernel_range):
        """
        随机方向和长度的线性运动模糊核。
        """
        kernel_size = random.choice(
            range(kernel_range[0], kernel_range[1] + 1, 2)
        )
        angle_rad = np.deg2rad(np.random.uniform(0, 180))
        center = kernel_size // 2

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cos_val, sin_val = np.cos(angle_rad), np.sin(angle_rad)
        for i in range(kernel_size):
            offset = i - center
            x = int(round(center + offset * cos_val))
            y = int(round(center + offset * sin_val))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        if kernel.sum() > 0:
            kernel /= kernel.sum()

        kernel_tensor = torch.FloatTensor(kernel).to(img_tensor.device)
        return filter2D(img_tensor, kernel_tensor)

    def _video_codec_compress(self, img_tensor, codecs, crf_range, presets):
        """
        用 ffmpeg 做单帧 H.264/H.265 编解码，
        产生真实的视频压缩伪影（块效应、振铃、色度损失）。
        """
        codec = random.choice(codecs)
        crf = random.randint(crf_range[0], crf_range[1])
        preset = random.choice(presets)

        frame_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)

        height, width = frame_np.shape[:2]
        height_even = height - height % 2
        width_even = width - width % 2
        frame_bgr = cv2.cvtColor(
            frame_np[:height_even, :width_even], cv2.COLOR_RGB2BGR
        )

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = os.path.join(tmpdir, 'compressed.mp4')

                subprocess.run(
                    [
                        'ffmpeg', '-y', '-loglevel', 'error',
                        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                        '-s', f'{width_even}x{height_even}',
                        '-i', 'pipe:0',
                        '-c:v', codec,
                        '-crf', str(crf),
                        '-preset', preset,
                        '-pix_fmt', 'yuv420p',
                        '-frames:v', '1',
                        output_path,
                    ],
                    input=frame_bgr.tobytes(),
                    capture_output=True,
                    timeout=15,
                )

                decode_proc = subprocess.run(
                    [
                        'ffmpeg', '-y', '-loglevel', 'error',
                        '-i', output_path,
                        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                        '-frames:v', '1',
                        'pipe:1',
                    ],
                    capture_output=True,
                    timeout=15,
                )

                if decode_proc.returncode == 0 and len(decode_proc.stdout) > 0:
                    decoded = np.frombuffer(
                        decode_proc.stdout, dtype=np.uint8
                    ).reshape(height_even, width_even, 3)
                    decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
                    decoded_tensor = torch.from_numpy(
                        decoded_rgb.astype(np.float32) / 255.0
                    ).permute(2, 0, 1).unsqueeze(0)
                    return decoded_tensor.to(img_tensor.device)
        except (subprocess.TimeoutExpired, Exception):
            pass

        return img_tensor

    @torch.no_grad()
    def video_screenshot_degrade_process(self, img_gt, resize_bak=False):
        """
        纯视频截图退化流程，独立于 degrade_process。

        模拟真实视频截图经历的退化链路：
          1. 下采样（视频分辨率通常低于原图）
          2. YUV420 色度下采样（视频标准 4:2:0）
          3. 可选的运动模糊（视频帧中常见）
          4. 视频编解码压缩（H.264/H.265，核心退化源）
          5. 可选的轻微噪声（传感器噪声 / 解码误差）
          6. 可选的再压缩（截图保存为 JPEG 时的二次压缩）
          7. 可选的色彩偏移（YUV/RGB 色彩空间转换损失）

        Args:
            img_gt: 输入 GT 图像 (numpy, [H,W,3], float32, 0-1)
            resize_bak: 是否 resize 回原始尺寸

        Returns:
            (img_gt, img_lq): GT tensor [1,C,H,W] 和退化后的 LQ tensor [1,C,H,W]
        """
        # numpy -> tensor
        img_gt = self.random_augment(img_gt)
        img_gt = img_gt.to(self.device)
        ori_h, ori_w = img_gt.size()[2:4]

        vopt = self.opt.get('video_screenshot', {})

        scale_final = self.opt['scale']
        if isinstance(scale_final, list):
            scale_final = random.randint(scale_final[0], scale_final[1])

        out = img_gt.clone()

        # ==================== Step 1: 下采样 ==================== #
        target_h = ori_h // scale_final
        target_w = ori_w // scale_final
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, size=(target_h, target_w), mode=mode)

        # ==================== Step 2: YUV420 色度下采样 ==================== #
        if np.random.uniform() < vopt.get('yuv420_prob', 0.8):
            out = self._yuv420_chroma_subsample(out)

        # ==================== Step 3: 运动模糊 ==================== #
        if np.random.uniform() < vopt.get('motion_blur_prob', 0.3):
            kernel_range = vopt.get('motion_blur_kernel_range', [5, 15])
            out = self._motion_blur(out, kernel_range)

        # ==================== Step 4: 视频编解码压缩 ==================== #
        if np.random.uniform() < vopt.get('codec_prob', 0.8):
            codecs = vopt.get('codecs', ['libx264', 'libx265'])
            crf_range = vopt.get('crf_range', [23, 38])
            presets = vopt.get('presets', ['ultrafast', 'fast', 'medium'])
            out = self._video_codec_compress(out, codecs, crf_range, presets)
        else:
            jpeg_range = vopt.get('jpeg_fallback_range', [30, 60])
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        # ==================== Step 5: 轻微噪声 ==================== #
        if np.random.uniform() < vopt.get('noise_prob', 0.4):
            noise_sigma_range = vopt.get('noise_sigma_range', [1, 10])
            out = random_add_gaussian_noise_pt(
                out, sigma_range=noise_sigma_range,
                clip=True, rounds=False, gray_prob=0.0)

        # ==================== Step 6: 截图二次 JPEG 压缩 ==================== #
        if np.random.uniform() < vopt.get('screenshot_compress_prob', 0.5):
            jpeg_range = vopt.get('screenshot_jpeg_range', [60, 95])
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*jpeg_range)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)

        # ==================== Step 7: 色彩偏移 ==================== #
        if np.random.uniform() < vopt.get('color_shift_prob', 0.3):
            brightness = vopt.get('brightness', (0.9, 1.1))
            contrast = vopt.get('contrast', (0.9, 1.1))
            saturation = vopt.get('saturation', (0.8, 1.2))
            hue = vopt.get('hue', (-0.02, 0.02))
            out = self.color_jitter_pt(out, brightness, contrast, saturation, hue)

        # ==================== resize back ==================== #
        if resize_bak:
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)

        img_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return img_gt, img_lq