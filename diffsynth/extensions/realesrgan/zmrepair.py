# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

RGB_TO_YUV = torch.tensor([
    [ 0.29900,  0.58700,  0.11400],
    [-0.14714119, -0.28886916,  0.43601035],
    [ 0.61497538, -0.51496512, -0.10001026],
], dtype=torch.float32)

YUV_TO_RGB = torch.tensor([
    [1.0,  0.0,      1.13983],
    [1.0, -0.39465, -0.58060],
    [1.0,  2.03211,  0.0    ],
], dtype=torch.float32)


# ── ZMLiveRepair network ──

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 3"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)


class DownConv(nn.Module):
    """Downscaling with avgpool then double/triple conv"""

    def __init__(self, in_channels, out_channels, num_conv=2):
        super().__init__()
        if num_conv == 2:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.pool_conv = nn.Sequential(
                nn.AvgPool2d(2),
                TripleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.pool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResnetBlock(nn.Module):
    """Define a mobile-version Resnet block"""

    def __init__(self, dim):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)
        return out


class UpCatConvResNet(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
            self.conv = DoubleConv(in_channels, out_channels)

        self.resblock1 = ResnetBlock(out_channels)
        self.subpixel = nn.PixelShuffle(2)

    def interpolate(self, x):
        tensor_temp = x
        for i in range(3):
            tensor_temp = torch.cat((tensor_temp, x), 1)
        x = tensor_temp
        x = self.subpixel(x)
        return x

    def forward(self, x1, x2):
        x1 = self.interpolate(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)

        ##Resnet
        out1 = self.resblock1(out)

        return out1


class UpCatConvResNet_np(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
            self.conv = DoubleConv(in_channels, out_channels)

        self.resblock1 = ResnetBlock(out_channels)
        self.subpixel = nn.PixelShuffle(2)

    def interpolate(self, x):
        tensor_temp = x
        for i in range(3):
            tensor_temp = torch.cat((tensor_temp, x), 1)
        x = tensor_temp
        x = self.subpixel(x)
        return x

    def forward(self, x1, x2):
        x1 = self.interpolate(x1)

        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)

        ##Resnet
        out1 = self.resblock1(out)

        return out1


class ZMLiveRepair(nn.Module):
    """UNetVEAlqv13_w2d_np Defines a U-Net video enhancement network

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch=1, ngf=4):
        super().__init__()

        self.padder_size = 16

        self.inconv1 = DoubleConv(num_in_ch, 3*ngf)
        # downsample
        self.down0 = DownConv(3*ngf, 5*ngf)

        self.d0_resnet = nn.Sequential(
            ResnetBlock(5*ngf),
        )
        self.down1 = DownConv(5*ngf, 9*ngf)

        self.d1_resnet = nn.Sequential(
            ResnetBlock(9*ngf),
        )

        self.down2 = DownConv(9*ngf, 16*ngf)

        self.d2_resnet = nn.Sequential(
            ResnetBlock(16*ngf),
        )
        self.down3 = DownConv(16*ngf, 28*ngf, num_conv=3)

        self.d3_resnet = nn.Sequential(
            ResnetBlock(28*ngf),
        )

        #resnet_up
        self.up3_resnet = UpCatConvResNet_np(44*ngf, 16*ngf)
        self.up2_resnet = UpCatConvResNet_np(25*ngf, 9*ngf)
        self.up1_resnet = UpCatConvResNet_np(14*ngf, 5*ngf)
        self.up0_resnet = UpCatConvResNet_np(8*ngf, 3*ngf)

        # extra convolutions
        self.outconv1 = nn.Conv2d(3*ngf, num_in_ch, 3, 1, 1, bias=False)

    def forward(self, x, alpha=1.0):
        x, H, W = self.check_image_size(x)

        x0 = self.inconv1(x)
        # downsample
        x1 = self.down0(x0)  # 1/2
        x1 = self.d0_resnet(x1)
        x2 = self.down1(x1)  # 1/4
        x2 = self.d1_resnet(x2)
        x3 = self.down2(x2)  # 1/8
        x3 = self.d2_resnet(x3)
        x4 = self.down3(x3)  # 1/16
        x4 = self.d3_resnet(x4)

        x5 = self.up3_resnet(x4, x3)  # 1/8
        x5 = self.up2_resnet(x5, x2)  # 1/4
        x5 = self.up1_resnet(x5, x1)  # 1/2
        x5 = self.up0_resnet(x5, x0)  # 1
        x5 = self.outconv1(x5)
        out = x + x5
        out = (1-alpha) * x + alpha * out
        return out[..., :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='reflect')
        return x, h, w


# ── ZMRepair wrapper ──

class ZMRepair:
    def __init__(self, model_path, device="cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = ZMLiveRepair()
        state_dict = torch.load(model_path, map_location="cpu")
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().requires_grad_(False).to(self.device, dtype=self.dtype)

        self.rgb_to_yuv = RGB_TO_YUV.to(self.device)
        self.yuv_to_rgb = YUV_TO_RGB.to(self.device)

    @torch.no_grad()
    def __call__(self, images, batch_size=10):
        """
        Args:
            images: [N, C, H, W], float32, [0, 1]
        Returns:
            restored: [N, C, H, W], float32, [0, 1]
        """
        images = images.to(self.device, dtype=self.dtype)
        ori_h, ori_w = images.shape[-2:]

        all_out = []
        for start in range(0, images.shape[0], batch_size):
            chunk = images[start:start + batch_size]
            yuv = torch.einsum("b c h w, c k -> b k h w", chunk, self.rgb_to_yuv.T)
            y_restored = self.model(yuv[:, :1, :, :])
            yuv_restored = torch.cat([y_restored, yuv[:, 1:, :, :]], dim=1)
            rgb_restored = torch.einsum("b c h w, c k -> b k h w", yuv_restored, self.yuv_to_rgb.T)
            rgb_restored = F.interpolate(rgb_restored, size=(ori_h, ori_w), mode="bicubic")
            all_out.append(rgb_restored.clip(0, 1))
        return torch.cat(all_out, dim=0)


# ── CLI ──

def read_image_paths_from_txt(txt_path):
    """从 txt 文件读取图片路径列表，每行一个路径，跳过空行"""
    paths = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                paths.append(line)
    return paths


def get_args():
    parser = argparse.ArgumentParser(description="ZMRepair: Y-channel image repair.")
    parser.add_argument("--model_path", type=str, default="/mnt/workspace/hrbai/ckpt-cache/zhimei/zhimei_live_repair_video_enhance_model.pth")
    parser.add_argument("--input_txt", type=str, required=True,
                        help="Path to a txt file listing image paths (one per line).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel worker processes. Each loads its own model copy.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start index (inclusive) to slice the image list from txt. Default: 0.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End index (exclusive) to slice the image list from txt. Default: len(list).")
    return parser.parse_args()


def worker_main(worker_id, num_workers, args, image_paths):
    total = len(image_paths)
    chunk_size = (total + num_workers - 1) // num_workers
    start_idx = worker_id * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    worker_paths = image_paths[start_idx:end_idx]

    if not worker_paths:
        print(f"[Worker {worker_id}] No images assigned, exiting.")
        return

    print(f"[Worker {worker_id}] Processing images [{start_idx}, {end_idx}) "
          f"({len(worker_paths)}/{total}) on {args.device}")

    restorer = ZMRepair(args.model_path, device=args.device)
    skipped = 0

    for i, path in enumerate(worker_paths):
        out_path = os.path.join(args.output_dir, os.path.basename(path))

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            restored = restorer(tensor, batch_size=1).cpu()
            out_np = (restored[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(out_np).save(out_path)
        except Exception as exc:
            print(f"[Worker {worker_id}] Failed on {path}: {exc}")

        if (i + 1) % 50 == 0 or (i + 1) == len(worker_paths):
            print(f"[Worker {worker_id}] {i + 1}/{len(worker_paths)} (skipped {skipped})")

    print(f"[Worker {worker_id}] Done. Total {len(worker_paths)}, skipped {skipped}.")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_paths = read_image_paths_from_txt(args.input_txt)
    print(f"Total entries in txt: {len(all_paths)}")

    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else len(all_paths)
    image_paths = all_paths[start:end]
    print(f"Selected range [{start}, {end}): {len(image_paths)} image(s)")

    if not image_paths:
        return

    num_workers = max(1, args.num_workers)

    if num_workers == 1:
        worker_main(0, 1, args, image_paths)
    else:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)

        processes = []
        for wid in range(num_workers):
            proc = mp.Process(target=worker_main, args=(wid, num_workers, args, image_paths))
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()

    print("All done!")


if __name__ == "__main__":
    main()
