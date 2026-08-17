# 数据预处理管线

基于 Qwen 系列 VLM 与 SAM 3，将电商商品组图处理为参考图引导超分辨率（Reference-based SR）训练/测试所需的数据集。

## 环境与变量准备

**Python 依赖**：`tqdm`、`numpy`、`scikit-learn`、`Pillow`、`opencv-python`、`dashscope` SDK。

**API Key**：在本目录下创建 `.env` 文件并配置：

```bash
DASHSCOPE_API_KEY=<你的DashScope API Key>
# 如使用 OpenAI 兼容接口，另需配置：
# OPENAI_API_KEY=<你的OpenAI API Key>
# OPENAI_BASE_URL=<你的OpenAI兼容接口地址>
```

**SAM 3 环境**（`sam3_pipe.py` 生成商品 mask 时使用）：安装好即可直接运行，要求 Python ≥ 3.12、PyTorch ≥ 2.7、CUDA ≥ 12.6 的 GPU：

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e .
```

模型权重需另行下载 SAM 3 checkpoint（`sam3.pt`），并在 `sam3_pipe.py` 中将 `checkpoint_path` 指向实际路径。

## 整体流程

```
商品组图（候选图像 + 商品元信息）
        │
        ▼
[阶段 1] annotate.py ──────── 去重/相关性过滤 → 参考主图优选 → 商品定位裁剪 → embedding 相似度
        │  输出: 每商品标注 JSON + annotation_list.json
        ▼
[阶段 2] 数据集转换
        ├── convert_to_train_dataset.py ── 训练集: hq_ori.txt / ref_crop.txt / prompt_ori.txt
        │                                  （短边 ≥ 512 且 HQ-REF 相似度 ≥ 0.85）
        └── convert_to_test_dataset.py ─── 测试集: HQ + 合成 LQ（Real-ESRGAN 退化）+ REF + prompt
        │
        ▼
[阶段 3] 可选后处理
        ├── align_ref_to_hq.py ──── 双图联合定位 + 仿射变换，REF 对齐到 HQ 的商品区域
        └── sam3_pipe.py ────────── SAM 3 文本提示分割，生成商品 mask（ref_mask.txt）
        │
        ▼
训练（train_edit_mask.sh）/ 测试（test_edit_mask.sh）
```

真实场景 benchmark 测试集由 `convert_csv_to_realworld_testset.py` 单独构建（CSV 清单 → 描述生成 → REF 对齐）。

## Script 执行命令

```bash
# 阶段 1：VLM 标注（annotate_test.sh 为小规模试跑，annotate_train.sh 为全量）
bash annotate_test.sh

# 阶段 2：标注 JSON → 训练集 / 测试集
bash convert_annotation_to_dataset.sh

# 阶段 3：REF 仿射对齐（可选，参数见脚本内 usage 示例）
python align_ref_to_hq.py --hq-txt ... --ref-txt ... --prompt-txt ... --output-dir ... --save-comparison

# 阶段 3：SAM 3 生成商品 mask（可选，输入/输出路径在脚本内配置）
python sam3_pipe.py

# 真实场景测试集构建（可选）
bash convert_csv_to_realworld_testset.sh
```

训练集三个 txt（`hq_ori.txt` / `ref_crop.txt` / `prompt_ori.txt`）及 mask 列表直接作为根目录 `train_edit_mask.sh` 的输入参数；测试集用于 `test_edit_mask.sh` 的各 `TEST_MODE`。
