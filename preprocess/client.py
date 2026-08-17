import json
import os
import re
import time
import logging
from typing import Any, Dict, List, Tuple, Optional, Literal, Union

import dashscope
import requests
from openai import OpenAI
from dotenv import load_dotenv
from http import HTTPStatus

logger = logging.getLogger(__name__)

MODEL_CNY_PRICES = {
    # LLM per 1000 tokens
    "qwen3-max": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
    "qwen3-max-2025-09-23": {"input": 0.006 / 1000, "output": 0.024 / 1000},
    "qwen3-max-2026-01-23": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
    "qwen-max": {"input": 0.0024 / 1000, "output": 0.0096 / 1000},
    "qwen-max-latest": {"input": 0.0024 / 1000, "output": 0.0096 / 1000},
    "qwen-max-2025-01-25": {"input": 0.0024 / 1000, "output": 0.0096 / 1000},
    "qwen3.5-plus": {"input": 0.0008 / 1000, "output": 0.0048 / 1000},
    "qwen3.5-plus-2026-02-15": {"input": 0.0008 / 1000, "output": 0.0048 / 1000},
    "qwen-plus": {"input": 0.0008 / 1000, "output": 0.002 / 1000},
    "qwen-plus-latest": {"input": 0.0008 / 1000, "output": 0.002 / 1000},
    "qwen-plus-2025-12-01": {"input": 0.0008 / 1000, "output": 0.002 / 1000},
    "qwen-turbo": {"input": 0.0003 / 1000, "output": 0.0006 / 1000},
    "qwen-turbo-latest": {"input": 0.0003 / 1000, "output": 0.0006 / 1000},
    "qwen3.5-flash": {"input": 0.0002 / 1000, "output": 0.002 / 1000},
    "qwen3.5-flash-2026-02-23": {"input": 0.0002 / 1000, "output": 0.002 / 1000},
    "qwen-flash": {"input": 0.00015 / 1000, "output": 0.0015 / 1000},
    "qwen-flash-2025-07-28": {"input": 0.00015 / 1000, "output": 0.0015 / 1000},
    # VLM per 1000 tokens
    "qwen3-vl-flash": {"input": 0.00015 / 1000, "output": 0.0015 / 1000},
    "qwen3-vl-flash-2025-10-15": {"input": 0.00015 / 1000, "output": 0.0015 / 1000},
    "qwen3-vl-plus": {"input": 0.001 / 1000, "output": 0.01 / 1000},
    "qwen3-vl-plus-2025-09-23": {"input": 0.001 / 1000, "output": 0.01 / 1000},
    "qwen-vl-max": {"input": 0.0016 / 1000, "output": 0.004 / 1000},
    "qwen-vl-max-latest": {"input": 0.0016 / 1000, "output": 0.004 / 1000},
    "qwen-vl-max-2025-08-13": {"input": 0.0016 / 1000, "output": 0.004 / 1000},
    "qwen3-omni-flash": {"input": 0.0158 / 1000, "output": 0.0127 / 1000},
    "qwen3-omni-flash-2025-12-01": {"input": 0.0158 / 1000, "output": 0.0127 / 1000},
    # Embedding per 1000 tokens
    "text-embedding-v4": {"input": 0.0005 / 1000, "output": 0},
    "qwen3-vl-embedding": {"input": 0.0018 / 1000, "output": 0},
    "qwen3-rerank": {"input": 0.0005 / 1000, "output": 0},
    # ASR per seconds
    "paraformer-v2": {"input": 0.00008, "output": 0},
    "paraformer-realtime-8k-v2": {"input": 0.00024, "output": 0},
    # T2I per image
    "qwen-image-2.0-pro": {"input": 0, "output": 0.5},
    "qwen-image-2.0": {"input": 0, "output": 0.2},
}

DEFAULT_DASHSCOPE_MODELS = [
    "qwen3.7-max", "qwen3.7-max-2026-05-20",
    "qwen3.6-max-preview",
    "qwen3.6-plus", "qwen3.6-plus-2026-04-02",
    "qwen3.6-flash", "qwen3.6-flash-2026-04-16",
    "qwen3.5-plus", "qwen3.5-plus-2026-02-15",
    "qwen3.5-flash", "qwen3.5-flash-2026-02-23",
    "qwen3.5-omni-plus", "qwen3-omni-flash",
    "qwen3-vl-plus", "qwen3-vl-plus-2025-09-23",
    "qwen3-vl-flash", "qwen3-vl-flash-2025-10-15",
    "qwen3-max", "qwen3-max-2025-09-23", "qwen3-max-2026-01-23",
    "qwen-vl-max", "qwen-vl-max-latest", "qwen-vl-max-2025-08-13",
    "qwen-vl-plus", "qwen-vl-plus-latest",
    "qwen-max", "qwen-max-latest", "qwen-max-2025-01-25",
    "qwen-plus", "qwen-plus-latest",
    "qwen-turbo", "qwen-turbo-latest", "qwen-flash",
    "qwen-image-2.0-pro", "qwen-image-2.0",
    "paraformer-realtime-8k-v2",
]


def get_model_price(model: str) -> Tuple[float, float]:
    """获取模型价格(CNY)"""
    model_price = MODEL_CNY_PRICES.get(model, {"input": 0, "output": 0})
    return model_price["input"], model_price["output"]


def get_retry_model(model: str) -> Optional[str]:
    """获取重试模型"""
    return None


def parse_json(text: str) -> Optional[Union[Dict, List]]:
    """
    从字符串中提取并解析JSON对象。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pattern = r"```json\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_string = match.group(1).strip()
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                return None

    return None


class Client:
    def __init__(self, api_provider: Literal['dashscope', 'ai_studio'] = 'dashscope',
                 dashscope_models: Optional[List[str]] = None):
        self.api_provider = api_provider
        self.dashscope_models = dashscope_models or DEFAULT_DASHSCOPE_MODELS

        load_dotenv()
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.dashscope_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.usages = {}
        self.total_costs = {}

    def get_usages(self, usage_id: str = "", reset: bool = False) -> []:
        """
        打印当前的模型用量并重置用量
        """
        current_usage = self.usages.get(usage_id, [])
        current_cost = self.total_costs.get(usage_id, 0.0)
        logger.info(f"Usages for {usage_id}: {current_usage}")
        logger.info(f'Total cost for {usage_id}: ¥{current_cost:.5f}')

        if reset:
            self.usages[usage_id] = []
            self.total_costs[usage_id] = 0
        return current_usage

    def format_user_content(self, prompt: str, model: str, video_url: Optional[str] = None, fps: float = 2.0,
                            audio_url: Optional[str] = None, image_url: Optional[Union[str, List[str]]] = None):
        """
        格式化用户消息内容
        """
        if model in self.dashscope_models:
            content = [{"text": prompt}]
            if video_url:
                # fps 可参数控制视频抽帧频率，表示每隔 1/fps 秒抽取一帧
                content.append({"video": video_url, "fps": fps}, )
            if image_url:
                if isinstance(image_url, list):
                    for i in image_url:
                        content.append({"image": i})
                else:
                    content.append({"image": image_url})
            if audio_url:
                content.append({"audio": audio_url})
        else:
            content = [{"type": "text", "text": prompt}, ]
            if video_url:
                content.append({"type": "video_url", "video_url": {"url": video_url}})
            if image_url:
                if isinstance(image_url, list):
                    for i in image_url:
                        content.append({"type": "image_url", "image_url": {"url": i}})
                else:
                    content.append({"type": "image_url", "image_url": {"url": image_url}})
            if audio_url:
                content.append({"type": "audio_url", "audio_url": {"url": audio_url}})
        return content

    def openai_chat(self, messages: List[Dict[str, Any]], model: str, json_mode: bool = False,
                    extra_body: Optional[Dict[str, Any]] = None, usage_id: Optional[str] = None) -> str:
        """
        通过openai sdk调用LLM
        """
        started_at = time.time()
        if model in self.dashscope_models:
            client = self.dashscope_client
        else:
            client = self.openai_client
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"} if json_mode else None,
            extra_body=extra_body,
        )
        ended_at = time.time()
        logger.info(f'{model} response (openai sdk) takes {(ended_at - started_at):.2f}s')

        if completion.usage:
            input_price, output_price = get_model_price(model)
            cost = completion.usage.prompt_tokens * input_price + completion.usage.completion_tokens * output_price
            if usage_id:
                self.usages.setdefault(usage_id, [])
                self.usages[usage_id].append({
                    "model": model, "cost": cost,
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                    "time": ended_at - started_at
                })
                self.total_costs.setdefault(usage_id, 0.0)
                self.total_costs[usage_id] += cost

            logger.info(f"Prompt tokens: {completion.usage.prompt_tokens}, "
                        f"Completion tokens: {completion.usage.completion_tokens}, "
                        f"Total tokens: {completion.usage.total_tokens}")
            logger.info(f"Cost: ¥{cost:.4f}")

        return completion.choices[0].message.content

    def dashscope_chat(self, messages: List[Dict[str, Any]], model: str,
                       json_mode: bool = False, enable_thinking: bool = False,
                       videos_cache_dir: Optional[str] = None, usage_id: Optional[str] = None) -> str:
        """
        通过dashscope sdk调用LLM
        """
        if not model.startswith("qwen"):
            raise ValueError("Only qwen models can be accessed through dashscope sdk")

        started_at = time.time()
        if '-vl-' in model or 'qwen3.5' in model or 'qwen3.6' in model:
            # 视觉大模型
            response = dashscope.MultiModalConversation.call(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                model=model,
                messages=messages,
                result_format='message',
                enable_thinking=enable_thinking
            )
            if response.status_code != HTTPStatus.OK:
                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS and get_retry_model(model):
                    retry_model = get_retry_model(model)
                    response = dashscope.MultiModalConversation.call(
                        api_key=os.getenv('DASHSCOPE_API_KEY'),
                        model=retry_model,
                        messages=messages,
                        result_format='message',
                        enable_thinking=enable_thinking
                    )
                    if response.status_code != HTTPStatus.OK:
                        raise RuntimeError(f"Failed to call {retry_model} (dashscope sdk): {response}")
                else:
                    raise RuntimeError(f"Failed to call {model} (dashscope sdk): {response}")
            output = response.output.choices[0].message.content[0]["text"]
        else:
            # 大语言模型
            response = dashscope.Generation.call(
                api_key=os.getenv('DASHSCOPE_API_KEY'),
                model=model,
                messages=messages,
                response_format={"type": "json_object"} if json_mode else {"type": "text"},
                result_format='message',
                enable_thinking=enable_thinking
            )
            if response.status_code != HTTPStatus.OK:
                if response.status_code == HTTPStatus.TOO_MANY_REQUESTS and get_retry_model(model):
                    retry_model = get_retry_model(model)
                    response = dashscope.Generation.call(
                        api_key=os.getenv('DASHSCOPE_API_KEY'),
                        model=retry_model,
                        messages=messages,
                        response_format={"type": "json_object"} if json_mode else {"type": "text"},
                        result_format='message',
                        enable_thinking=enable_thinking
                    )
                    if response.status_code != HTTPStatus.OK:
                        raise RuntimeError(f"Failed to call {retry_model} (dashscope sdk): {response}")
                elif response.message == "Output data may contain inappropriate content.":
                    response = dashscope.Generation.call(
                        api_key=os.getenv('DASHSCOPE_API_KEY'),
                        model=model,
                        messages=messages,
                        response_format={"type": "json_object"} if json_mode else {"type": "text"},
                        result_format='message',
                        enable_thinking=enable_thinking
                    )
                    if response.status_code != HTTPStatus.OK:
                        raise RuntimeError(f"Failed to call {model} (dashscope sdk) twice: {response}")
                else:
                    raise RuntimeError(f"Failed to call {model} (dashscope sdk): {response}")
            output = response.output.choices[0].message.content
        ended_at = time.time()

        input_price, output_price = get_model_price(model)
        cost = response.usage.input_tokens * input_price + response.usage.output_tokens * output_price
        if usage_id:
            self.usages.setdefault(usage_id, [])
            self.usages[usage_id].append({
                "model": model, "cost": cost,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "time": ended_at - started_at
            })
            self.total_costs.setdefault(usage_id, 0.0)
            self.total_costs[usage_id] += cost

        logger.info(f'{model} response (dashscope sdk) takes {(ended_at - started_at):.2f}s')
        logger.info(f"Prompt tokens: {response.usage.input_tokens}, "
                    f"Completion tokens: {response.usage.output_tokens}, "
                    f"Total tokens: {response.usage.total_tokens}")
        logger.info(f"Cost: ¥{cost:.4f}")

        return output

    def chat(self, messages: List[Dict[str, Any]], model: str, json_mode: bool = False,
             enable_thinking: bool = False, extra_body: Optional[Dict[str, Any]] = None,
             videos_cache_dir: Optional[str] = None, usage_id: Optional[str] = None) -> str:
        """
        调用LLM
        """
        if model in self.dashscope_models:
            return self.dashscope_chat(messages=messages, model=model, json_mode=json_mode,
                                       enable_thinking=enable_thinking, videos_cache_dir=videos_cache_dir,
                                       usage_id=usage_id)
        else:
            return self.openai_chat(messages=messages, model=model, json_mode=json_mode,
                                    extra_body=extra_body, usage_id=usage_id)

    def multi_modal_embed(self, inputs: list[Dict[str, Any]], model: str = "qwen3-vl-embedding",
                          dimension: int = 1024, usage_id: Optional[str] = None) -> list:
        """
        通过dashscope sdk调用embedding
        """
        started_at = time.time()
        response = dashscope.MultiModalEmbedding.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=model,
            input=inputs,
            dimension=dimension,  # 指定向量维度
            output_type="dense"
        )
        ended_at = time.time()

        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Embedding API failed: {response}")

        input_price, output_price = get_model_price(model)
        total_tokens = response.usage["total_tokens"] if isinstance(response.usage, dict) else response.usage.total_tokens
        cost = total_tokens * input_price
        if usage_id:
            self.usages.setdefault(usage_id, [])
            self.usages[usage_id].append({
                "model": model, "cost": cost,
                "total_tokens": total_tokens,
                "time": ended_at - started_at
            })
            self.total_costs.setdefault(usage_id, 0.0)
            self.total_costs[usage_id] += cost

        logger.info(f'{model} response (dashscope sdk) takes {(ended_at - started_at):.2f}s')
        logger.info(f"Token tokens: {total_tokens}")
        logger.info(f"Cost: ¥{cost:.4f}")

        return [e['embedding'] for e in response.output['embeddings']]

    def convert_to_json(self, content: str, json_schema: Any, model: str = "qwen-flash",
                        usage_id: Optional[str] = None) -> Dict[str, Any]:
        """
        调用llm从文本中提取json
        """
        user_prompt = f"""请从一段文本中提取结构化JSON信息，这是你必须遵循的格式规范（JSON Schema）：
        {json_schema}

        这是需要处理的原始文本：
        {content}

        现在，请根据以上规范和文本，提取对应的 JSON 数据：
        """
        messages = [
            {
                "role": "user",
                "content": self.format_user_content(user_prompt, model=model)
            }
        ]
        response = self.chat(messages=messages, model=model, json_mode=False,
                             extra_body={"enable_thinking": False} if model.startswith('qwen') else None,
                             usage_id=usage_id)
        result = parse_json(response)
        return result

    def generate_image_dashscope(self, prompt: str, image_path: str, model: str = "qwen-image-2.0-pro",
                                 size: str = "2048*2048", usage_id: Optional[str] = None):
        """
        调用llm编辑图片
        """
        started_at = time.time()
        messages = [
            {
                "role": "user",
                "content": self.format_user_content(prompt, model=model, image_url=image_path)
            }
        ]
        response = dashscope.MultiModalConversation.call(
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model=model,
            messages=messages,
            stream=False,
            n=1,
            watermark=False,
            negative_prompt=" ",
            prompt_extend=True,
            size=size,
        )
        ended_at = time.time()

        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Failed to call {model} (dashscope sdk): {response}")

        for i, content in enumerate(response.output.choices[0].message.content):
            logger.info(f"Image {i}: {content['image']}")

        input_price, output_price = get_model_price(model)
        cost = response.usage.image_count * output_price
        if usage_id:
            self.usages.setdefault(usage_id, [])
            self.usages[usage_id].append({
                "model": model, "cost": cost,
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "image_count": response.usage.image_count,
                "time": ended_at - started_at
            })
            self.total_costs.setdefault(usage_id, 0.0)
            self.total_costs[usage_id] += cost

        logger.info(f'{model} response (dashscope sdk) takes {(ended_at - started_at):.2f}s')
        logger.info(f"Prompt tokens: {response.usage.input_tokens}, "
                    f"Completion tokens: {response.usage.output_tokens}, "
                    f"Image count: {response.usage.image_count}")
        logger.info(f"Cost: ¥{cost:.4f}")

        return response.output.choices[0].message.content[0]['image']

    def generate_image_gemini(self, prompt: str, image_path: str,
                              model: str = "gemini-3.1-flash-image-preview"):
        """
        调用llm编辑图片
        """
        import base64
        import mimetypes
        import io
        from PIL import Image

        base_url = f'https://idealab.alibaba-inc.com/api/vertex/v1beta/models/{model}:generateContent'
        headers = {
            'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}',
            'Content-Type': 'application/json'
        }

        # local file path
        base64str = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        mimetype = mimetypes.guess_type(image_path)[0]
        if mimetype is None:
            mimetype = 'image/png'
            buffer = io.BytesIO()
            Image.open(image_path).save(buffer, format="PNG")
            base64str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inlineData": {
                                "mimeType": mimetype,
                                "data": base64str
                            }
                        },
                    ],
                },
            ],
            "generationConfig": {
                # "seed": number,
                # "topP": number,
                # "topK": number,
                # "candidateCount": integer,
                # "maxOutputTokens": integer,
                # "presencePenalty": float,
                # "frequencyPenalty": float,
                # "stopSequences": [
                #     string
                # ],
                # "responseMimeType": string,
                # "responseSchema": schema,
                # "responseLogprobs": boolean,
                # "logprobs": integer,
                # "audioTimestamp": boolean,
                # "thinkingConfig": {
                #     "thinkingBudget": integer,
                #     "thinkingLevel": enum
                # },
                # "mediaResolution": enum
                "imageConfig": {
                    "aspectRatio": "1:1",
                    "imageSize": "2K"
                }
            },
        }
        response = requests.post(base_url, headers=headers, json=data)

        text_response = ''
        image_response = None
        if response.status_code != 200:
            logger.error(response.text)
        else:
            try:
                res = json.loads(response.content)
                texts = []
                images = []
                for part in res['candidates'][0]['content']['parts']:
                    if part['text']:
                        texts.append(part['text'])
                    if part['inlineData']:
                        base64str = part['inlineData']['data']
                        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64str, "utf-8"))))
                        images.append(img)
                text_response = '\n'.join(texts)
                image_response = images[0]
            except Exception as e:
                logger.error(e)
        return image_response
