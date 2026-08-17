FILTER_IMAGES_PROMPT_TEMPLATE = """## 任务
请基于输入的候选图像集，完成以下工作：
1. **去重与筛选**：剔除商品形态及画面内容重复，以及未展示同一挂链目标商品（如展示其他品类、款式、SKU）的图像；
2. **商品描述**：为该挂链目标商品提供准确的视觉描述。

## 规则
1. **去重标准**：保留彼此之间存在实际拍摄视觉内容差异的图像，剔除仅因裁剪、缩放、填充、加字、修图、宽高比调整等后期修改造成的重复。
2. **相关标准**：所有保留的图像必须展示**同一件挂链目标商品**，即商品在颜色、纹理、款式等上完全一致。
2. **剔除依据**：是否剔除仅关注“是否展示同一目标商品”以及“是否有实质性内容差异”。严禁基于图像质量和商品展示状态等进行剔除，这些因素仅用于在两张重复图像之间进行比较选择更好的一张。
3. **禁止保留**：所有多图拼接或宫格形式的组合图像（即一张图内包含多个子图）必须被剔除，无论其内容是否相关。。

## 输入内容
- 挂链商品元信息（如品类、名称等）；  
- 一组按序编号的候选图像。

## 输出要求
请严格按以下的Markdown格式输出：

### 第一部分：去重分析（不含 JSON）
- 说明去重与筛选过程；
- 列出被剔除图像的编号及其原因（依据去重以及相关标准）；
- 明确是否至少有一张图像展示了挂链目标商品。

### 第二部分：结构化结果（用```json ...```包裹）
{
    "product_displayed": true/false,
    "product_name": "挂链目标商品的简洁名称，仅包含视觉描述，不含营销用语",
    "product_entity": "不含任何描述性前后缀的挂链目标商品主体词，如连衣裙，T恤",
    "product_description": "详细描述这件商品的视觉特征（如颜色、材质、纹理、款式、设计等）"
    "unqualified_images": [
        {"number": 被剔除图像编号, "reason": "具体剔除原因"},
        ...
    ],
    "qualified_numbers": [保留的去重以及相关图像编号列表]
}
"""

PICK_REFERENCE_IMAGE_PROMPT_TEMPLATE = """## 任务
请基于输入的候选图像集，完成以下工作：
1. 从中选出最适合作为商品参考主图的一张，并说明理由；
2. 保留与参考主图存在实际拍摄视觉内容差异（如形态、姿势、角度、位置等）的图像，剔除因裁剪、缩放、填充、加字、修图、宽高比调整等后期修改造成的重复图像。

## 参考主图评估标准
1. **主体完整性与呈现度**
   商品应作为画面中唯一或主导的视觉主体，完整入镜且轮廓清晰：
   - 整体形态及关键识别特征（如品牌标识、结构、纹理）清晰可见；
   - 占据充足的视觉比重，居中或显著突出；
   - 非局部特写，无裁切（不被画面边缘截断）；
   - 无严重遮挡（如被手、头发、包装膜、文字贴图等大面积覆盖）。

2. **展示规范性与状态真实性**
   商品须处于原始销售状态，真实反映其本体属性：
   - 未使用、未变形、未沾污、未涂抹；
   - 形态完整，符合出厂或销售标准；
   - 避免展示使用过程、效果或人为改造状态。

3. **姿态与角度合理性**
   商品应以利于识别的方式稳定呈现：
   - 采用正面角度全面展示，避免背面、极端倾斜（如压扁、侧立成线）导致关键信息缺失；
   - 处于静止或准静止状态，无褶皱、扭曲、折叠、缠绕等中间形变；
   - 无运动拖影或模糊，确保形态清晰可辨。

4. **画面洁净度与稳定性**
   背景与拍摄质量应支持商品清晰呈现：
   - 背景干净简洁，无杂乱物品、干扰性文字或图案；
   - 画面稳定，无明显抖动、快速平移、频繁变焦或剧烈晃动；
   - 光照均匀，无过曝、欠曝或强烈反光影响商品细节识别。

## 输入内容
- 挂链商品的视觉描述；
- 一组按序编号的候选图像。

## 输出要求
请严格按以下的Markdown格式输出：

### 第一部分：主图优选分析（不含 JSON）
- 对比所有合格图像，依据上述四大标准逐项分析优劣；
- 详细论证理由，明确指出哪张图像最符合参考主图要求；
- 依次检查出所有与参考主图内容实际拍摄视觉内容一致的重复图像。

### 第二部分：结构化结果（用```json ...```包裹）
{
   "pick_reason": "选择该图像的具体理由，需结合四大评估标准进行说明",
   "reference_image_number": 整数（最终选定的参考主图编号）,
   "duplicate_images": [
      {"number": 与参考主图实际拍摄视觉内容一致的重复图像编号, "reason": "具体重复分析"},
      ...
   ],
}
"""

OBJECT_DETECTION_KEYS = ["product_elements", "face_elements", "body_elements"]

OBJECT_DETECTION_PROMPT_TEMPLATE = """## 任务：
分析电商图像，识别并定位其中用于**智能裁剪和构图优化**的关键画面元素。裁剪构图的核心目标是完整且突出地展示实物商品，应尽可能包含商品的全部，避免生硬地截断商品的任意部分。

## 你将获得的信息输入：
1. 挂链商品元信息。
2. 完整的源画面图像。

## 输出指示：
以JSON格式输出裁剪构图时应考虑的关键元素：
- "description": 详细描述图像的完整视觉内容，包括构图组成，商品视觉特征（如颜色、材质、纹理、款式、设计等），任何文字信息等细节；
- "product_elements": 如有，列出正在展示的实物挂链商品（画面主体），否则为空，每个包括：
    - "bbox_2d": 边界框坐标,
    - "label": 商品名称,
    - "texture_point": 给出采样一张大块商品材质/纹理贴图的最佳中心坐标点。
- "face_elements": [如有完整可见的主播/模特人脸，列出并标明它们的边界框坐标（bbox_2d），名称（label），否则为空。]
- "body_elements": [如有完整可见的主播/模特躯干，列出并标明它们的主播/模特躯干，标明边界框坐标（bbox_2d），名称（label），否则为空。]
"""

VERIFY_IMAGE_CONSISTENCY_PROMPT_TEMPLATE = """## 任务
你将收到两张图像：
- **原始参考图像**：包含挂链目标商品的真实拍摄图；
- **生成参考图像**：由AI模型基于原始图像生成的1:1白底商品图。

请严格判断：**生成图像中的商品是否在所有关键视觉属性上与原始图像中的挂链目标商品完全一致**。

## 判断标准（必须全部满足才视为“一致”）
1. **品类与实体一致**：商品属于同一实体类型（如均为“女式圆领短袖T恤”，不能是“T恤” vs “背心”）。
2. **款式结构一致**：领型、袖长、下摆、开衩、口袋、拉链、纽扣等结构细节完全相同。
3. **颜色与图案一致**：主色、辅色、印花、条纹、logo位置及内容、图案分布等无任何偏差。
4. **材质与纹理一致**：面料质感（如棉、丝绸、牛仔）、反光特性、织纹等在视觉上应匹配。
5. **无新增或缺失元素**：生成图不得添加原图没有的装饰、标签、水印，也不得遗漏原图中存在的关键部件（如吊牌、品牌标、特殊缝线）。
6. **比例与形态一致**：商品整体比例、松紧度、垂感、褶皱自然状态应与原图一致，不得因AI生成导致形变（如袖子变宽、衣身拉长等）。

## 输入内容
- 挂链商品元信息（如品类、名称等）；
- 原始参考图像（编号 #original）；
- 生成参考图像（编号 #generated）。

## 输出要求
请严格按以下 Markdown 格式输出：

### 第一部分：一致性分析
- 逐项对照上述6条标准，说明生成图与原始图是否一致；
- 若存在不一致，明确指出具体差异点（如“原图为蓝色条纹，生成图为纯蓝色”）；
- 最终结论：是否完全一致。

### 第二部分：结构化结果（用```json ...```包裹）
{
    "inconsistency_details": "若不一致，简明列出所有差异；若一致，填空字符串",
    "is_consistent": true/false,
    "confidence": "高/中/低"
}
"""

IMAGE_CAPTION_PROMPT = """请用一段话详细地描述这张高质量图像的所有视觉内容（如构图布局，画面元素，文字图案，纹理细节等）"""
PRODUCT_CAPTION_PROMPT = """请用一段话详细地描述展示主商品的所有视觉特征（如颜色、材质、纹理、款式、设计、细节、图案等）"""

PAIRED_GROUNDING_PROMPT = """## Task
You will receive two images depicting the same product and its associated outfit:
- **First image (SRC)**: a scene/model image;
- **Second image (REF)**: a high-quality reference image of the same product (possibly a white-background image, a flat-lay image, or a different viewing angle).

Localize the corresponding region of the same product and its associated outfit in both images, and return the matched semantic bounding boxes for affine alignment.

## Key requirements
1. **Semantic consistency**: the two boxes must enclose the same visible part of the same product and its associated outfit. If SRC shows only the upper part of the product, the REF box should cover only the corresponding upper part rather than the whole product.
2. **Completeness**: under semantic consistency, each box should cover the product's visible region as completely as possible.

## Output format
Output strictly as a JSON object (wrapped in ```json ...```):
{
    "src_image_description": "Describe the full visual content of the first image (SRC) in detail, including composition layout, visual elements, product visual attributes, text/patterns, and any texture details",
    "product_entity": "The plain object noun of the product without any descriptive prefix/suffix, e.g., dress, T-shirt",
    "product_description": "Describe the visual attributes of the product in detail (e.g., color, material, texture, style, design)",
    "match_description": "The matched semantic region of the product and its associated outfit in the two images",
    "src_matched_bbox": "Bounding box [x1, y1, x2, y2] of the matched semantic region in the SRC image",
    "ref_matched_bbox": "Bounding box [x1, y1, x2, y2] of the matched semantic region in the REF image"
}
"""

# ======================== VLM 评测 prompt（真实场景超分 pointwise 打分） ========================

EVALUATION_SYSTEM_PROMPT = """You are a rigorous, objective image-quality evaluator for reference-based super-resolution. For each image you get a global view and a full-resolution crop of the key region: use the global view for content/composition and the crop for fine detail/texture. Return only a valid JSON object. Each score is an integer 1-5 (1=worst, 5=best), and the full range should be used."""

EVALUATION_TASK_PROMPT = """## Task
Evaluate a real-world image restoration result. The inputs, each consisting of a global view and a key-region crop, are:
- **PRED**: the model output to evaluate;
- **REF**: the reference showing the true textures of the key item.

Grade PRED on its intrinsic quality and its consistency with REF along the dimensions below, and return the scores in the specified JSON format.

## Evaluation dimensions (the key goal is restoring correct, faithful texture of the key item)
1. **Perceptual quality**: overall visual quality --- sharpness, natural realistic details, correct tone/color, and absence of degradation. Penalize blur, insufficient enhancement, noise, artifacts, over-smoothing, over-sharpening halos, over-saturation, and color shift. Note that real high-resolution images naturally contain fine high-frequency textures; a visibly sharper yet faithful result should score higher, and only false details (halos, ringing, hallucinated textures) are penalized.
2. **Texture consistency**: whether the key item's textures/patterns match the REF, and whether they blend seamlessly in lighting, perspective, and style. Penalize destroyed, fabricated, or mismatched textures.

## Output format
Output strictly as a JSON object (wrapped in ```json ...```):
{
    "perceptual_quality": integer score 1-5 for the perceptual quality of PRED,
    "texture_consistency": integer score 1-5 for the texture consistency between PRED and REF
}
"""

TILE_DESCRIPTION_ENGLISH_PROMPT_TEMPLATE = """## Task
You will receive a series of image patches (tiles) cropped from a single image, numbered sequentially starting from 1. The image may contain a variety of content — products, people, backgrounds, text, or other objects.

For each tile, provide a concise visual content description that captures:
1. **Semantic content**: What is shown in this tile (e.g., a product's collar/sleeve/hemline, a person's face/hand/hair, plain background, text/watermark, floor/wall, or other objects).
2. **Texture and material**: Any visible surface texture, fabric weave, skin texture, hair detail, surface finish, sheen, or material characteristics.
3. **Color and pattern**: Dominant colors, gradients, prints, stripes, skin tones, or other visual patterns.
4. **Other details**: Any notable details such as stitching, buttons, wrinkles, folds, lighting conditions, bokeh, noise, or compression artifacts.

## Rules
- Each description should be 1-3 sentences, focused and specific to what is actually visible in that tile.
- If a tile contains only background (e.g., plain white/gray/colored), describe it as background with its color and texture.
- If a tile contains text, watermarks, or logos, mention them and describe their content.
- If a tile contains a person or body parts, describe the visible features (skin, hair, clothing, pose, etc.).
- Descriptions should be useful as conditioning prompts for image super-resolution, so emphasize visual and textural details.

## Output Format
Output strictly as a JSON object (wrapped in ```json ...```):
{
    "tile_descriptions": [
        {
            "tile_number": 1,
            "description": "Visual content description of tile 1"
        },
        {
            "tile_number": 2,
            "description": "Visual content description of tile 2"
        }
    ]
}
"""