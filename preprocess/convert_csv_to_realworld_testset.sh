#!/bin/bash

CSV_PATH='/mnt/workspace/yihu/projects/ODTSR/dataset/测试集捞取-20260416.csv'
OUTPUT_DIR='/mnt/workspace/yihu/projects/ODTSR/dataset/realworld_testset_260416_with_opt_prompt'

python convert_csv_to_realworld_testset.py \
    --csv "${CSV_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --grounding-model "qwen3.5-plus" \
    --caption-model "qwen3.5-plus" \
    --lossless

echo "Script Executed."
