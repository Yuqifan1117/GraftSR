#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 合成 benchmark 评估（有参考 + 无参考指标，结果与 HQ 按文件名配对）
python evaluate_synthetic.py \
        --output_dir /root/your/path/experiments/your_exp_name/benchmark_synthetic_degraded-250_scale_4_fidelity-1.0 \
        --gt_dir /root/your/path/benchmark/synthetic_250/hq_ori \
        --metrics psnr,ssim,lpips,dists,niqe,musiq,clipiqa,maniqa-pipal \
        --crop_border 4 \
        --save_path /root/your/path/experiments/summary/your_exp_name_synthetic250_metrics.json
