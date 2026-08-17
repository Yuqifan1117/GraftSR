#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# 真实场景 benchmark 评估（无参考指标；提供 gt_dir 时也可计算有参考指标）
python evaluate_real.py \
        --output_dir /root/your/path/experiments/your_exp_name/benchmark_realworld-50_scale_4.0_fidelity-1.0 \
        --gt_dir /root/your/path/benchmark/realworld/lq_ori \
        --metrics niqe,musiq,clipiqa,maniqa-pipal \
        --crop_border 0 \
        --save_path /root/your/path/experiments/summary/your_exp_name_realworld_metrics.json
