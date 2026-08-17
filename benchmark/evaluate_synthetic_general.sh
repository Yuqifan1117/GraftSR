#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 通用/公开 benchmark 评估（RealSR、DRealSR 等，复用 evaluate_synthetic.py）
python evaluate_synthetic.py \
        --output_dir /root/your/path/experiments/your_exp_name/benchmark_drealsr_scale_4_fidelity-1.0 \
        --gt_dir /root/your/path/open_benchmarks/datasets/benchmark_drealsr/test_HR \
        --metrics psnr,ssim,lpips,dists,niqe,musiq,clipiqa,maniqa-pipal \
        --crop_border 4 \
        --save_path /root/your/path/experiments/summary/your_exp_name_drealsr_metrics.json
