#!/bin/bash

python annotate.py \
    --metadata-list-json '/mnt/workspace/hrbai/data/CDXQ/odps_download_2k_60w_preprocess_download_info_sample10.json' \
    --candidate-image-dir '/mnt/workspace/hrbai/data/CDXQ/images' \
    --main-image-dir '/mnt/workspace/yihu/projects/ODTSR/dataset/main_images' \
    --output-dir '/mnt/workspace/yihu/projects/ODTSR/annotation/260324_1950_loop5_qwen3.5-plus_qwen3.5-plus' \
    --annotation-list-json '/mnt/workspace/yihu/projects/ODTSR/annotation/260324_1950_loop5_qwen3.5-plus_qwen3.5-plus/annotation_list.json' \
    --vlm-filter-model 'qwen3.5-plus' \
    --vlm-select-model 'qwen3.5-plus' \
    --vlm-detect-model 'qwen3.5-plus' \
    --image-edit-model 'qwen-image-2.0' \
    --loop-filter-iterations '5' \
    --num-workers '2'

# --enable-image-generation

echo "Script Executed."