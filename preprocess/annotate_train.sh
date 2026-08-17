#!/bin/bash

ANNOTATION_DIR='/mnt/workspace/yihu/projects/ODTSR/dataset/cdxq_60w_260331_all-qwen3.5-plus/annotation'
ANNOTATION_LIST_JSON='/mnt/workspace/yihu/projects/ODTSR/dataset/cdxq_60w_260331_all-qwen3.5-plus/annotation_list.json'

# first batch: /mnt/workspace/hrbai/data/CDXQ/odps_download_2k_60w_preprocess_download_info.json
# second batch: /mnt/workspace/hrbai/data/CDXQ/odps_download_from_40w_to_91w_preprocess_deduplicate_download_request.json

python annotate.py \
    --metadata-list-json '/mnt/workspace/hrbai/data/CDXQ/odps_download_2k_60w_preprocess_download_info.json' \
    --candidate-image-dir '/mnt/workspace/hrbai/data/CDXQ/images' \
    --main-image-dir '/mnt/workspace/yihu/projects/ODTSR/dataset/main_images' \
    --output-dir "$ANNOTATION_DIR" \
    --annotation-list-json "$ANNOTATION_LIST_JSON" \
    --vlm-filter-model 'qwen3.6-plus' \
    --vlm-select-model 'qwen3.6-plus' \
    --vlm-detect-model 'qwen3.6-plus' \
    --loop-filter-iterations '5' \
    --num-workers '12' \
    --log-level 'WARNING'

echo "Script Executed."