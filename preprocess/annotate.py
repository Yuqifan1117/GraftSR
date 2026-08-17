import argparse
import json
import os
import shutil
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from client import Client, parse_json
from image_utils import (resize_vlm_elements,
                         download_image,
                         compress_image,
                         crop_image,
                         pil_image_to_base64,
                         get_image_extension)
from prompts import (OBJECT_DETECTION_KEYS,
                     OBJECT_DETECTION_PROMPT_TEMPLATE,
                     FILTER_IMAGES_PROMPT_TEMPLATE,
                     PICK_REFERENCE_IMAGE_PROMPT_TEMPLATE, VERIFY_IMAGE_CONSISTENCY_PROMPT_TEMPLATE)

logger = logging.getLogger(__name__)


class Annotator:
    def __init__(self, client: Client):
        self.client = client

    def filter_images(self, product_id: str,
                      product_metadata: Dict[str, Any],
                      candidate_image_paths: List[str],
                      model: str = "qwen3.5-plus") -> Dict[str, Any]:
        """
        第一阶段：去重与筛选
        剔除重复或未展示同一商品的图像
        """
        user_prompt = FILTER_IMAGES_PROMPT_TEMPLATE + f"""
        ## 挂链目标商品信息：
        {product_metadata}
        """
        messages = [{
            "role": "user",
            "content": self.client.format_user_content(user_prompt, model=model,
                                                       image_url=candidate_image_paths)
        }]
        response = self.client.chat(messages=messages, model=model, json_mode=False, usage_id=product_id)
        result = parse_json(response)
        result["original_response"] = response
        return result

    def select_reference_image(self, product_id: str,
                               product_name: str,
                               candidate_image_paths: List[str],
                               model: str = "qwen3.5-plus") -> Dict[str, Any]:
        """
        第二阶段：主图优选
        从筛选后的图像中选择最适合作为参考主图的一张
        """
        user_prompt = PICK_REFERENCE_IMAGE_PROMPT_TEMPLATE + f"""
        ## 挂链目标商品的视觉描述：
        {product_name}
        """
        messages = [{
            "role": "user",
            "content": self.client.format_user_content(user_prompt, model=model,
                                                       image_url=candidate_image_paths)
        }]
        response = self.client.chat(messages=messages, model=model, json_mode=False, usage_id=product_id)
        result = parse_json(response)
        result["original_response"] = response
        return result

    def merge_loop_filter_results(self, all_filter_results: List[Dict[str, Any]],
                                  candidate_image_paths: List[str]) -> Dict[str, Any]:
        """
        合并所有迭代的过滤结果
        """
        final_filter_results = {}

        # 合并所有迭代的 unqualified_images，并将编号映射回原始编号
        all_unqualified_images = []

        # 用于追踪从原始图像到当前迭代图像的映射关系
        # iteration_mappings[i] 表示第 i 次迭代时，当前图像编号对应的原始图像编号
        iteration_mappings = []
        current_to_original_mapping = list(range(len(candidate_image_paths)))

        for iteration in range(len(all_filter_results)):
            iteration_mappings.append(current_to_original_mapping.copy())

            filter_results = all_filter_results[iteration]
            unqualified_images = filter_results.get("unqualified_images", [])

            # 将当前迭代中的不合格图像编号映射回原始编号
            for unq_img in unqualified_images:
                current_number = unq_img.get("number") - 1
                if 0 <= current_number < len(current_to_original_mapping):
                    original_number = current_to_original_mapping[current_number] + 1
                    unq_img["number"] = original_number
                    all_unqualified_images.append(unq_img)

            # 更新映射关系，只保留合格图像
            qualified_numbers = filter_results.get("qualified_numbers", [])
            if qualified_numbers:
                qualified_indexes = [i - 1 for i in qualified_numbers]
                current_to_original_mapping = [current_to_original_mapping[i] for i in qualified_indexes]

        # 使用最后一次迭代的商品信息
        if all_filter_results:
            last_iteration = all_filter_results[-1]
            final_filter_results["product_displayed"] = last_iteration.get("product_displayed", True)
            final_filter_results["product_name"] = last_iteration.get("product_name", "")
            final_filter_results["product_entity"] = last_iteration.get("product_entity", "")
            final_filter_results["product_description"] = last_iteration.get("product_description", "")

        # 合并所有不合格图像
        final_filter_results["unqualified_images"] = all_unqualified_images

        # 最终合格图像的编号（映射回原始编号）
        final_qualified_numbers = [num + 1 for num in current_to_original_mapping]
        final_filter_results["qualified_numbers"] = final_qualified_numbers

        logger.debug(f"Merged filter results: {len(all_unqualified_images)} unqualified images, "
                     f"{len(final_qualified_numbers)} qualified images")
        return final_filter_results

    def merge_select_results(self, filter_results: Dict[str, Any],
                             select_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并第一阶段和第二阶段的结果
        """

        qualified_numbers = filter_results.get("qualified_numbers")
        qualified_indexes = [i - 1 for i in qualified_numbers]

        final_select_results = {}

        # 合并第一阶段和第二阶段的 unqualified_images
        unqualified_images = filter_results.get("unqualified_images", []).copy()
        duplicate_images = select_results.get("duplicate_images", [])

        # 将第二阶段的 duplicate_images 映射回原始编号并合并到 unqualified_images
        if duplicate_images:
            for dup_img in duplicate_images:
                dup_index = dup_img.get("number") - 1
                if 0 <= dup_index < len(qualified_indexes):
                    original_number = qualified_indexes[dup_index] + 1
                    dup_img["number"] = original_number
            unqualified_images.extend(duplicate_images)

        final_select_results["unqualified_images"] = unqualified_images

        # 从 qualified_numbers 中排除 duplicate_images
        final_qualified_numbers = filter_results.get("qualified_numbers", [])
        duplicate_original_numbers = [dup_img["number"] for dup_img in duplicate_images]
        final_qualified_numbers = [num for num in final_qualified_numbers if num not in duplicate_original_numbers]
        final_select_results["qualified_numbers"] = final_qualified_numbers

        # 将选定的图像编号映射回原始图像编号
        selected_index = select_results.get("reference_image_number", 1) - 1
        if selected_index < 0 or selected_index >= len(qualified_indexes):
            raise ValueError(
                f"Invalid reference_image_number: {selected_index + 1} for {len(qualified_indexes)} qualified images")
        original_index = qualified_indexes[selected_index]
        final_select_results["reference_image_number"] = original_index + 1

        return final_select_results

    def save_selected_images(self, final_select_results: Dict[str, Any],
                             candidate_image_paths: List[str],
                             output_dir: str,
                             product_id: str):
        """
        保存选定的参考主图、合格和不合格的图像
        """

        # 保存选定的参考主图
        ref_number = final_select_results.get("reference_image_number")
        if not isinstance(ref_number, int) or ref_number - 1 < 0 or ref_number - 1 >= len(candidate_image_paths):
            raise ValueError(f"Invalid reference_image_number: {ref_number} for {len(candidate_image_paths)} images")
        reference_image_path = candidate_image_paths[ref_number - 1]
        reference_ext = get_image_extension(reference_image_path)
        output_image_path = os.path.join(output_dir, f"{product_id}_ref{reference_ext}")
        shutil.copy(reference_image_path, output_image_path)

        # 分开保存合格的和不合格的图像
        final_select_results["qualified_image_paths"] = []
        final_select_results["unqualified_image_paths"] = []
        qualified_numbers = (final_select_results.get("qualified_numbers") or
                             [item["number"] for item in final_select_results["qualified_images"]])
        qualified_indexes = [i - 1 for i in qualified_numbers]
        for i, image_path in enumerate(candidate_image_paths):
            filename = os.path.basename(image_path)
            if i in qualified_indexes:
                # 如果是参考主图，添加 _ref 后缀
                if i == ref_number - 1:
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_ref{ext}"
                    dst = os.path.join(output_dir, filename)
                    final_select_results["reference_image_path"] = dst
                else:
                    name, ext = os.path.splitext(filename)
                    filename = f"{name}_qualified{ext}"
                    dst = os.path.join(output_dir, filename)
                    final_select_results["qualified_image_paths"].append(dst)

                shutil.copy(image_path, dst)
            else:
                name, ext = os.path.splitext(filename)
                filename = f"{name}_unqualified{ext}"
                dst = os.path.join(output_dir, filename)
                final_select_results["unqualified_image_paths"].append(dst)

                shutil.copy(image_path, dst)

        return output_image_path

    def spatial_grounding(self, product_id: str,
                          product_name: str,
                          image_path: str,
                          model: str = "qwen3.5-flash"):
        """
        空间定位，用于裁剪商品
        """
        # 将图像缩放到1080p再送入qwen vl
        img = Image.open(image_path)
        width, height = img.size
        canvas_width = 1080
        canvas_height = int(canvas_width * height / width)
        if width != canvas_width:
            img = img.resize((canvas_width, canvas_height), Image.Resampling.BICUBIC)
        image_base64 = pil_image_to_base64(img)
        resize_version = "qwen3vl" if model.startswith("qwen3") else "qwen25vl"

        user_prompt = OBJECT_DETECTION_PROMPT_TEMPLATE + f"""
        ## 已知商品信息：
        {product_name}
        """
        messages = [{
            "role": "user",
            "content": self.client.format_user_content(user_prompt, model=model, image_url=image_base64)
        }]
        response = self.client.chat(messages=messages, model=model, json_mode=False, usage_id=product_id)
        result = parse_json(response)
        if result is None:
            # TODO: 传递模型参数
            result = self.client.convert_to_json(content=response, json_schema="合法的JSON对象", usage_id=product_id)

        # 将边界框转换回原图坐标系
        result = resize_vlm_elements(result=result, elements_keys=OBJECT_DETECTION_KEYS,
                                     input_image_height=canvas_height, input_image_width=canvas_width,
                                     original_image_height=height, original_image_width=width,
                                     version=resize_version)

        return result

    def save_grounded_images(self, grounding_result: Dict[str, Any],
                             original_image_path: str):
        """
        保存空间定位后的裁剪图像
        """
        product_elements = grounding_result.get("product_elements", [])
        if product_elements and len(product_elements) > 0:
            original_ext = get_image_extension(original_image_path)
            # 使用第一个商品的 bbox 进行裁剪
            bbox = product_elements[0].get("bbox_2d", [])
            cropped_image_path = original_image_path.replace(original_ext, "_cropped.png")
            crop_image(bbox, original_image_path, cropped_image_path, lossless=True)
        else:
            cropped_image_path = None

        return cropped_image_path

    def calculate_selection_similarity(self, product_id: str,
                                       reference_image_path: str,
                                       qualified_image_paths: list[str],
                                       batch_size: int = 5):
        """
        计算合格图像与参考图像的相似度
        """
        # 构造输入格式
        all_image_paths = [reference_image_path] + qualified_image_paths
        compressed_image_paths = [compress_image(img_path, max_size_mb=4.9) for img_path in all_image_paths]
        embed_inputs = [{"image": img_path} for img_path in compressed_image_paths]

        # 获取所有图像的embedding
        all_embeddings = []
        for i in range(0, len(embed_inputs), batch_size):
            batch = embed_inputs[i:i + batch_size]
            logger.debug(f"Embedding batch {i // batch_size + 1}: {len(batch)} items")
            batch_embeddings = self.client.multi_modal_embed(
                inputs=batch,
                model="qwen3-vl-embedding",
                dimension=1024,
                usage_id=product_id
            )
            all_embeddings.extend(batch_embeddings)

        for ori_img_path, compressed_img_path in zip(all_image_paths, compressed_image_paths):
            if ori_img_path != compressed_img_path and os.path.exists(compressed_img_path):
                os.remove(compressed_img_path)
                logger.debug(f"Removed temporary compressed image: {compressed_img_path}")

        # 参考图像的embedding是第一个
        reference_embedding = np.array(all_embeddings[0]).reshape(1, -1)

        # 计算合格图像与参考图像的相似度
        qualified_similarities = {}
        for i, img_path in enumerate(qualified_image_paths, start=1):
            img_embedding = np.array(all_embeddings[i]).reshape(1, -1)
            similarity = cosine_similarity(reference_embedding, img_embedding)[0][0]
            qualified_similarities[img_path] = float(similarity)

        return qualified_similarities

    def generate_reference_image(self, product_id: str,
                                 product_entity: str,
                                 image_path: str,
                                 model: str = "qwen-image-2.0"):
        """
        生成参考主图
        """
        # 压缩图像以确保不超过 10MB 限制
        compressed_image_path = compress_image(image_path, max_size_mb=10)
        
        prompt = f"""提取图中的{product_entity}，作为一张只有该商品的纯净白底图，保证商品完全一致，图像比例为1:1"""
        image_response = self.client.generate_image_dashscope(
            prompt=prompt,
            image_path=compressed_image_path,
            model=model,
            usage_id=product_id
        )
        
        # 如果创建了压缩文件，删除临时文件
        if compressed_image_path != image_path and os.path.exists(compressed_image_path):
            os.remove(compressed_image_path)
            logger.debug(f"Removed temporary compressed image: {compressed_image_path}")
        
        return image_response

    def verify_edit_consistency(self, product_id: str,
                                product_entity: str,
                                original_image_path: str,
                                edited_image_path: str,
                                model: str = "qwen3.5-plus",):
        """
        验证编辑后的图像是否一致
        """
        user_prompt = VERIFY_IMAGE_CONSISTENCY_PROMPT_TEMPLATE + f"""
        ## 挂链目标商品：
        {product_entity}
        """
        messages = [{
            "role": "user",
            "content": self.client.format_user_content(user_prompt, model=model,
                                                       image_url=[original_image_path, edited_image_path])
        }]
        response = self.client.chat(messages=messages, model=model, json_mode=False, usage_id=product_id)
        result = parse_json(response)
        result["original_response"] = response
        return result

    def run_pipeline(self, product_metadata: Dict[str, Any],
                     main_image_path: Optional[str],
                     candidate_image_paths: List[str],
                     output_json_path: str,
                     vlm_filter_model: str = "qwen3.5-plus",
                     vlm_select_model: str = "qwen3.5-plus",
                     vlm_detect_model: str = "qwen3.5-flash",
                     image_edit_model: str = "qwen-image-2.0",
                     loop_filter_iterations: int = 1,
                     enable_image_generation: bool = False,
                     ) -> Dict[str, Any]:
        if os.path.exists(output_json_path):
            with open(output_json_path, "r") as f:
                results = json.load(f)
            backup_json_path = output_json_path.replace(".json", "_backup.json")
            with open(backup_json_path, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
        else:
            results = {
                "product_metadata": product_metadata
            }
        output_dir = os.path.dirname(output_json_path)
        product_id = os.path.splitext(os.path.basename(output_json_path))[0]
        self.client.get_usages(usage_id=product_id, reset=True)

        # 步骤1: 图像选择

        # 第一阶段：去重与筛选
        if "all_filter_results" not in results and "final_filter_results" not in results:
            all_filter_results = []
            current_image_paths = candidate_image_paths
            qualified_image_paths = []
            for iteration in range(loop_filter_iterations):
                filter_results = self.filter_images(product_id,
                                                    product_metadata,
                                                    current_image_paths,
                                                    model=vlm_filter_model)
                # 获取筛选后的合格图像
                unqualified_images = filter_results.get("unqualified_images", [])
                qualified_numbers = filter_results.get("qualified_numbers", [])
                if len(unqualified_images) + len(qualified_numbers) < len(current_image_paths):
                    # 重试
                    logger.debug(f"Retrying iteration {iteration}")
                    continue
                if not qualified_numbers:
                    break
                qualified_indexes = [i - 1 for i in qualified_numbers]
                qualified_image_paths = [current_image_paths[i] for i in qualified_indexes]

                all_filter_results.append(filter_results)
                logger.debug(f"Filter results after iteration {iteration}: {filter_results}")

                if len(qualified_image_paths) <= 1 or iteration == loop_filter_iterations - 1:
                    break
                if len(qualified_image_paths) == len(current_image_paths) and len(qualified_image_paths) <= 5:
                    break
                current_image_paths = qualified_image_paths.copy()

            results["all_filter_results"] = all_filter_results
            logger.debug(f"All filter results: {all_filter_results}")

            if not qualified_image_paths or len(qualified_image_paths) < 2:
                raise ValueError(f"Only {len(qualified_image_paths) if qualified_image_paths else 0} qualified images after filtering")

            # 合并所有循环后的过滤结果
            final_filter_results = self.merge_loop_filter_results(all_filter_results,
                                                                  candidate_image_paths)
            results["final_filter_results"] = final_filter_results
            logger.debug(f"Final filter results: {final_filter_results}")

            if not final_filter_results.get("product_displayed") is True:
                raise ValueError("No relevant images displayed")
        else:
            final_filter_results = results["final_filter_results"]
            qualified_image_paths = [candidate_image_paths[i - 1] for i in final_filter_results.get("qualified_numbers", [])]

        # 第二阶段：从筛选后的图像中选择主图
        if "select_results" not in results and "final_select_results" not in results:
            product_name = final_filter_results.get("product_name", "")
            product_entity = final_filter_results.get("product_entity", "")
            select_results = self.select_reference_image(product_id,
                                                         product_name,
                                                         qualified_image_paths,
                                                         model=vlm_select_model)
            results["select_results"] = select_results
            logger.debug(f"Selected reference image: {select_results}")

            # 合并两阶段的结果
            final_select_results = self.merge_select_results(final_filter_results, select_results)
            results["final_select_results"] = final_select_results
            logger.debug(f"Final select results: {final_select_results}")
        else:
            final_select_results = results["final_select_results"]
            product_name = final_select_results.get("product_name", "")
            product_entity = final_select_results.get("product_entity", "")

        # 保存选定的参考图、合格和不合格的图像
        reference_image_path = self.save_selected_images(
            final_select_results,
            candidate_image_paths,
            output_dir,
            product_id
        )

        # 步骤2: 空间定位
        if "reference_grounding_result" not in results:
            reference_grounding_result = self.spatial_grounding(product_id,
                                                                product_name,
                                                                reference_image_path,
                                                                model=vlm_detect_model)
            results["reference_grounding_result"] = reference_grounding_result
            logger.debug(f"Reference grounding results: {reference_grounding_result}")

            # 保存裁剪后的参考图
            cropped_reference_path = self.save_grounded_images(
                reference_grounding_result,
                reference_image_path,
            )
            results["cropped_reference_image_path"] = cropped_reference_path

        if "qualified_grounding_results" not in results:
            qualified_grounding_results = []
            for qualified_image_path in final_select_results.get("qualified_image_paths", []):
                qualified_grounding_result = self.spatial_grounding(product_id,
                                                                    product_name,
                                                                    qualified_image_path,
                                                                    model=vlm_detect_model)
                qualified_grounding_results.append(qualified_grounding_result)

                # 保存裁剪后的合格图像
                cropped_qualified_path = self.save_grounded_images(
                    qualified_grounding_result,
                    qualified_image_path,
                )
                qualified_grounding_result["cropped_image_path"] = cropped_qualified_path
            results["qualified_grounding_results"] = qualified_grounding_results
            logger.debug(f"Qualified grounding results: {qualified_grounding_results}")

        # 步骤3: 基于embedding计算每张合格图像与参考图像的相似度
        if "qualified_similarities" not in results:
            qualified_similarities = self.calculate_selection_similarity(
                product_id,
                final_select_results.get("reference_image_path"),
                final_select_results.get("qualified_image_paths", [])
            )
            results["qualified_similarities"] = qualified_similarities
            logger.debug(f"Similarities between reference image and qualified images: {qualified_similarities}")

        # 步骤4: 生成参考图像（可选）
        if enable_image_generation and "edit_reference_image_path" not in results:
            edit_image_url = self.generate_reference_image(product_id,
                                                           product_entity,
                                                           reference_image_path,
                                                           model=image_edit_model)
            edit_image_path = os.path.join(output_dir, f"{product_id}_edited.png")
            download_image(edit_image_url, edit_image_path)
            results["edit_reference_image_path"] = edit_image_path

            verify_results = self.verify_edit_consistency(product_id,
                                                          product_entity,
                                                          reference_image_path,
                                                          edit_image_path,
                                                          model=vlm_select_model)
            results["edit_consistency_verification"] = verify_results
            logger.debug(f"Edit consistency verification: {verify_results}")
            if verify_results.get("is_consistent") is False:
                # 为编辑图像添加失败后缀
                failed_edit_image_path = os.path.join(output_dir, f"{product_id}_edited_failed.png")
                os.rename(edit_image_path, failed_edit_image_path)
                results["edit_reference_image_path"] = failed_edit_image_path

        if "model_usages" not in results:
            results["model_usages"] = self.client.get_usages(usage_id=product_id)

        with open(output_json_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        return results


def process_single_product(metadata: Dict[str, Any],
                           args: argparse.Namespace,
                           annotator: Annotator) -> Optional[Dict[str, Any]]:
    """
    处理单个商品的标注任务
    """
    product_id = metadata["item_id"]
    
    request_item_info = json.loads(metadata["request_item_info"])
    product_metadata = {
        "title": request_item_info["title"],
    }

    main_image_path = None
    if args.main_image_dir:
        main_image_path = os.path.join(args.main_image_dir, f"{product_id}.jpg")
        if not os.path.exists(main_image_path):
            main_image_path = None

    candidate_image_filenames = metadata["img_url_list"]
    candidate_image_paths = [os.path.join(args.candidate_image_dir, filename)
                             for filename in candidate_image_filenames]

    saved_dir = os.path.join(args.output_dir, f"{product_id}")
    os.makedirs(saved_dir, exist_ok=True)
    output_json_path = os.path.join(saved_dir, f"{product_id}.json")

    try:
        results = annotator.run_pipeline(product_metadata=product_metadata,
                                         main_image_path=main_image_path,
                                         candidate_image_paths=candidate_image_paths,
                                         output_json_path=output_json_path,
                                         vlm_filter_model=args.vlm_filter_model,
                                         vlm_select_model=args.vlm_select_model,
                                         vlm_detect_model=args.vlm_detect_model,
                                         image_edit_model=args.image_edit_model,
                                         loop_filter_iterations=args.loop_filter_iterations,
                                         enable_image_generation=args.enable_image_generation)

        annotation = {
            **metadata,
            "annotation": {
                "product_entity": results["final_filter_results"].get("product_entity"),
                "product_description": results["final_filter_results"].get("product_description"),
                "qualified_image_paths": results["final_select_results"].get("qualified_image_paths"),
                "reference_image_path": results["final_select_results"].get("reference_image_path"),
                "cropped_reference_image_path": results.get("cropped_reference_image_path"),
                "reference_grounding_result": results["reference_grounding_result"],
                "qualified_grounding_results": results["qualified_grounding_results"],
                "qualified_similarities": results["qualified_similarities"],
            }
        }
        return annotation
    except Exception as e:
        logger.error(f"Error processing product ID: {product_id}")
        logger.error(e)
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "product_id": product_id
        }
        with open(output_json_path, "w") as f:
            json.dump(error_info, f, ensure_ascii=False, indent=4)
        return None


def main(args):
    # 配置日志等级
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    client = Client(api_provider="dashscope")
    annotator = Annotator(client)

    with open(args.metadata_list_json, "r") as f:
        all_metadata_list = json.load(f)
    metadata_list = all_metadata_list[args.start:args.end]

    os.makedirs(args.output_dir, exist_ok=True)

    annotation_list = []
    num_workers = getattr(args, 'num_workers', 4)
    
    logger.info(f"Starting concurrent annotation with {num_workers} workers for {len(metadata_list)} products")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_product, metadata, args, annotator): metadata 
                   for metadata in metadata_list}
        
        with tqdm(total=len(metadata_list), desc="Processing products", unit="product") as pbar:
            for future in as_completed(futures):
                metadata = futures[future]
                product_id = metadata["item_id"]
                
                try:
                    result = future.result()
                    if result is not None:
                        annotation_list.append(result)
                        pbar.set_postfix({"product_id": product_id, "success": "✓"})
                    else:
                        pbar.set_postfix({"product_id": product_id, "success": "✗"})
                except Exception as e:
                    logger.error(f"Error processing product ID: {product_id}")
                    logger.error(e)
                    pbar.set_postfix({"product_id": product_id, "success": "✗"})
                
                pbar.update(1)
                
                if len(annotation_list) % 100 == 0 and args.annotation_list_json:
                    with open(args.annotation_list_json, "w") as f:
                        json.dump(annotation_list, f, ensure_ascii=False, indent=4)

    if args.annotation_list_json:
        with open(args.annotation_list_json, "w") as f:
            json.dump(annotation_list, f, ensure_ascii=False, indent=4)
    
    logger.info(f"Annotation completed. Successfully processed {len(annotation_list)}/{len(metadata_list)} products")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-list-json", type=str, required=True)
    parser.add_argument("--candidate-image-dir", type=str, required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--main-image-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--annotation-list-json", type=str, default=None)

    parser.add_argument("--vlm-filter-model", type=str, default="qwen3.5-plus")
    parser.add_argument("--vlm-select-model", type=str, default="qwen3.5-plus")
    parser.add_argument("--vlm-detect-model", type=str, default="qwen3.5-flash")
    parser.add_argument("--image-edit-model", type=str, default="qwen-image-2.0")
    parser.add_argument("--loop-filter-iterations", type=int, default=1)
    parser.add_argument("--enable-image-generation", action="store_true", help="Generate reference image")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of concurrent workers for annotation")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")

    args = parser.parse_args()
    main(args)
