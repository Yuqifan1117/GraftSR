"""
Differentiable Post-Processing Module

将 naturalize_folder.py 中的手工后处理操作转化为可微的 PyTorch 模块，
使 contrast/saturation/sharpness 参数可通过 MUSIQ + GT loss 联合优化。

设计原则：
1. 仅 3 个可学习标量参数，不可能过拟合
2. 所有操作完全可微，支持梯度回传
3. 与 PIL ImageEnhance 行为一致，保证训练起点 ≈ 手工后处理效果
4. 接口兼容 NaturalizationRefineHead，可直接替换
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiablePostProcess(nn.Module):
    """
    可微后处理模块：用 3 个可学习标量参数替代 CNN RefineHead。

    优化目标：
        min_theta  L(theta) = -w_m * MUSIQ(f_theta(x))
                               + w_p * MSE(f_theta(x), y)
                               + w_l * LPIPS(f_theta(x), y)

    Args:
        init_contrast: 对比度初始值（默认 1.02，与 naturalize_folder.py 一致）
        init_saturation: 饱和度初始值（默认 1.03）
        init_sharpness: 锐度初始值（默认 1.05）
        input_range: 输入 tensor 范围，'tanh'=[-1,1] 或 'sigmoid'=[0,1]
        clamp_params: 是否限制参数在合理范围内（防止优化跑飞）
        param_min: 参数下界
        param_max: 参数上界
    """

    def __init__(
        self,
        init_contrast=1.02,
        init_saturation=1.03,
        init_sharpness=1.05,
        input_range='tanh',
        clamp_params=True,
        param_min=0.8,
        param_max=1.3,
    ):
        super().__init__()
        self.input_range = input_range
        self.clamp_params = clamp_params
        self.param_min = param_min
        self.param_max = param_max

        # 3 个可学习标量参数
        self.contrast = nn.Parameter(torch.tensor(float(init_contrast)))
        self.saturation = nn.Parameter(torch.tensor(float(init_saturation)))
        self.sharpness = nn.Parameter(torch.tensor(float(init_sharpness)))

        # 预计算锐化核（Laplacian-based unsharp mask）
        # PIL Sharpness 等价于: out = img * factor + blurred * (1 - factor)
        # 其中 blurred 是 3x3 mean filter
        # 这里用可微卷积实现
        self.register_buffer(
            'blur_kernel',
            torch.ones(1, 1, 3, 3) / 9.0  # 3x3 mean filter
        )

    def _get_params(self):
        """获取 clamp 后的参数值"""
        if self.clamp_params:
            return (
                self.contrast.clamp(self.param_min, self.param_max),
                self.saturation.clamp(self.param_min, self.param_max),
                self.sharpness.clamp(self.param_min, self.param_max),
            )
        return self.contrast, self.saturation, self.sharpness

    @staticmethod
    def _apply_contrast(rgb, factor):
        """
        可微对比度调整，等价于 PIL ImageEnhance.Contrast。
        out = (in - mean) * factor + mean
        """
        mean = rgb.mean(dim=(2, 3), keepdim=True)  # (B, 3, 1, 1)
        return (rgb - mean) * factor + mean

    @staticmethod
    def _apply_saturation(rgb, factor):
        """
        可微饱和度调整，等价于 PIL ImageEnhance.Color。
        在 RGB 空间中：out = gray + (rgb - gray) * factor
        其中 gray = 0.299*R + 0.587*G + 0.114*B
        """
        weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
        gray = (rgb * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        return gray + (rgb - gray) * factor

    def _apply_sharpness(self, rgb, factor):
        """
        可微锐度调整，等价于 PIL ImageEnhance.Sharpness。
        out = rgb * factor + blur(rgb) * (1 - factor)
        """
        batch_size, channels, height, width = rgb.shape
        # 对每个通道独立做 3x3 mean blur
        rgb_flat = rgb.view(batch_size * channels, 1, height, width)
        blurred = F.conv2d(rgb_flat, self.blur_kernel, padding=1)
        blurred = blurred.view(batch_size, channels, height, width)
        return rgb * factor + blurred * (1.0 - factor)

    def forward(self, rgb):
        """
        Args:
            rgb: (B, 3, H, W) tensor
                 若 input_range='tanh'，范围为 [-1, 1]
                 若 input_range='sigmoid'，范围为 [0, 1]
        Returns:
            refined_rgb: 同尺寸、同范围的增强后 tensor
        """
        # 转到 [0, 1] 进行后处理（与 PIL 行为一致）
        if self.input_range == 'tanh':
            normalized = (rgb + 1.0) / 2.0
        else:
            normalized = rgb

        contrast, saturation, sharpness = self._get_params()

        # 依次应用三种后处理（顺序与 naturalize_folder.py 一致）
        out = self._apply_contrast(normalized, contrast)
        out = self._apply_saturation(out, saturation)
        out = self._apply_sharpness(out, sharpness)

        # clamp 到合法范围
        out = out.clamp(0.0, 1.0)

        # 转回原始范围
        if self.input_range == 'tanh':
            out = out * 2.0 - 1.0

        return out

    def get_num_parameters(self):
        """返回可训练参数数量（始终为 3）"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_current_params(self):
        """返回当前参数值（用于日志/调试）"""
        contrast, saturation, sharpness = self._get_params()
        return {
            'contrast': contrast.item(),
            'saturation': saturation.item(),
            'sharpness': sharpness.item(),
        }
