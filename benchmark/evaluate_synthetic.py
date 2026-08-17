"""
合成图像超分评估脚本（有参考 + 无参考）
用法：
    python evaluate_synthetic.py \
        --output_dir /path/to/enhanced_results \
        --gt_dir /path/to/benchmark/hq \
        --metrics psnr,ssim,lpips,niqe,musiq,clipiqa \
        --crop_border 4 \
        --save_path ./eval_results/synthetic_metrics.json

说明：
    - output_dir: 增强后的结果图片目录
    - gt_dir: 合成数据集对应的 HQ ground truth 目录
    - 两个目录中图片按文件名（不含扩展名）自动配对
"""

import sys
import os
import argparse
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "qwen_image"))

from metrics import run_standalone_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced images against HQ ground truth (synthetic)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="增强后结果图片所在目录")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="HQ ground truth 图片所在目录")
    parser.add_argument("--metrics", type=str,
                        default="psnr,ssim,lpips,niqe,musiq,clipiqa",
                        help="要计算的指标，逗号分隔。"
                             "有参考: psnr,ssim,lpips,dists | "
                             "无参考: niqe,musiq,clipiqa,maniqa | "
                             "数据集级别: fid")
    parser.add_argument("--crop_border", type=int, default=4,
                        help="PSNR/SSIM/NIQE 计算时裁剪的边界像素数（合成数据常用4）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--lpips_net", type=str, default="alex",
                        choices=["alex", "vgg"],
                        help="LPIPS 使用的骨干网络")
    parser.add_argument("--save_path", type=str, default=None,
                        help="结果保存路径，默认为 output_dir/metrics.json")
    args = parser.parse_args()

    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    print("=" * 60)
    print("Synthetic Image Super-Resolution Evaluation")
    print("=" * 60)
    print(f"  Output dir : {args.output_dir}")
    print(f"  GT dir     : {args.gt_dir}")
    print(f"  Metrics    : {metrics_list}")
    print(f"  Crop border: {args.crop_border}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    run_standalone_evaluation(
        output_dir=args.output_dir,
        gt_dir=args.gt_dir,
        metrics=metrics_list,
        crop_border=args.crop_border,
        device=args.device,
        lpips_net=args.lpips_net,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()