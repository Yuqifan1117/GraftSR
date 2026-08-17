"""
真实图像超分评估脚本
用法：
    python evaluate_real.py \
        --output_dir /data/oss_bucket_0/Users/yuqifan/test_outputs/qwen_one_step_gan/20260416-162357/net_gen_iter_12001/benchmark_realworld-50_ref-crop-flex-768_out-2073600_fidelity-1 \
        --gt_dir /data/oss_bucket_0/Users/yuqifan/ODTSR+/benchmark/realworld/lq_ori \
        --metrics psnr,ssim,lpips,niqe,musiq,clipiqa \
        --crop_border 0 \
        --save_path /data/oss_bucket_0/Users/yuqifan/test_outputs/qwen_one_step_gan/20260416-162357/net_gen_iter_12001/benchmark_realworld-50_ref-crop-flex-768_out-2073600_fidelity-1/metrics.json
"""

import sys
import os
os.environ['HF_ENDPOINT']="https://hf-mirror.com"
import argparse

# 将 metrics.py 所在目录加入搜索路径（benchmark/ 与 examples/qwen_image/ 的相对位置）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "qwen_image"))

from metrics import run_standalone_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced images against benchmark LQ")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="增强后结果图片所在目录")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="Benchmark LQ 图片所在目录（作为参考）")
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,niqe,musiq,clipiqa",
                        help="要计算的指标，逗号分隔。"
                             "有参考: psnr,ssim,lpips,dists | "
                             "无参考: niqe,musiq,clipiqa,maniqa | "
                             "数据集级别: fid")
    parser.add_argument("--crop_border", type=int, default=0,
                        help="PSNR/SSIM/NIQE 计算时裁剪的边界像素数")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--save_path", type=str, default=None,
                        help="结果保存路径，默认为 output_dir/metrics.json")
    args = parser.parse_args()

    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    run_standalone_evaluation(
        output_dir=args.output_dir,
        gt_dir=args.gt_dir,
        metrics=metrics_list,
        crop_border=args.crop_border,
        device=args.device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()