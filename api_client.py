import requests
import json
import base64
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import io
import numpy as np

from .config import (ZHIPU_CHAT_ENDPOINT, ZHIPU_IMAGE_ENDPOINT, ZHIPU_VIDEO_ENDPOINT, 
                    ALL_CHAT_MODELS, FREE_IMAGE_MODELS, FREE_VIDEO_MODELS)


class ZhipuAPIClient:
    """智谱AI API客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def image_to_base64(self, image: Union[Image.Image, np.ndarray]) -> str:
        """将图片转换为base64编码"""
        if isinstance(image, np.ndarray):
            # 将numpy数组转换为PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # 转换为base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_data = buffer.getvalue()
        return base64.b64encode(img_data).decode()
    
    def prepare_message_content(self, text: str, image: Optional[Union[Image.Image, np.ndarray]] = None) -> List[Dict[str, Any]]:
        """准备消息内容，支持文本和图片"""
        content = []
        
        # 添加文本内容
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        # 添加图片内容
        if image is not None:
            image_base64 = self.image_to_base64(image)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }
            })
        
        return content
    
    def chat_completion(self,
                       messages: List[Dict[str, Any]],
                       model: str = "glm-4.5",
                       temperature: float = 0.7,
                       max_tokens: int = 1024,
                       top_p: float = 0.9,
                       stream: bool = False,
                       **kwargs) -> Dict[str, Any]:
        """
        调用智谱AI对话补全接口
        
        Args:
            messages: 对话消息列表
            model: 使用的模型名称
            temperature: 温度参数 (0-1)
            max_tokens: 最大输出token数
            top_p: 核采样参数
            stream: 是否流式输出
            **kwargs: 其他参数
        
        Returns:
            API响应结果
        """
        if model not in ALL_CHAT_MODELS:
            raise ValueError(f"不支持的对话模型: {model}. 支持的模型: {ALL_CHAT_MODELS}")
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        # 参数兼容：部分思考类/视觉模型要求使用 max_new_tokens
        if isinstance(max_tokens, int) and max_tokens > 0:
            if "thinking" in model or model.endswith("-thinking-flash") or model.startswith("glm-4.1v"):
                payload["max_new_tokens"] = max_tokens
            else:
                payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                ZHIPU_CHAT_ENDPOINT,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            if response.status_code >= 400:
                raise Exception(f"API调用失败: {response.status_code} {response.text}")
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API调用失败: {str(e)}")
    
    def simple_chat(self, 
                   prompt: str, 
                   image: Optional[Union[Image.Image, np.ndarray]] = None,
                   model: str = "glm-4.5",
                   temperature: float = 0.7,
                   max_tokens: int = 1024,
                   system_prompt: str = "") -> str:
        """
        简单的对话接口
        
        Args:
            prompt: 用户输入的文本
            image: 可选的图片输入
            model: 使用的模型
            temperature: 温度参数
            max_tokens: 最大token数
            system_prompt: 系统提示词，将作为 role=system 注入
        
        Returns:
            AI的回复文本
        """
        # 准备消息内容（用户消息，可能包含文本+图片）
        content = self.prepare_message_content(prompt, image)
        
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        
        # 调用API
        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 提取回复内容
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API返回格式异常: {response}")
    
    def generate_image(self,
                      prompt: str,
                      model: str = "cogview-3-flash",
                      size: str = "1024x1024",
                      n: int = 1,
                      quality: str = "standard",
                      **kwargs) -> Dict[str, Any]:
        """
        调用智谱AI图片生成接口
        
        Args:
            prompt: 图片描述提示词
            model: 使用的图片生成模型
            size: 图片尺寸 (1024x1024, 768x1344, 1344x768等)
            n: 生成图片数量
            quality: 图片质量 (standard, hd)
            **kwargs: 其他参数
        
        Returns:
            API响应结果，包含生成的图片URL
        """
        if model not in FREE_IMAGE_MODELS:
            raise ValueError(f"不支持的图片生成模型: {model}. 支持的模型: {FREE_IMAGE_MODELS}")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
            "quality": quality,
            **kwargs
        }
        
        try:
            response = requests.post(
                ZHIPU_IMAGE_ENDPOINT,
                headers=self.headers,
                json=payload,
                timeout=120  # 图片生成需要更长时间
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"图片生成API调用失败: {str(e)}")
    
    def generate_video(self,
                      prompt: str,
                      model: str = "cogvideox-flash",
                      image_url: Optional[str] = None,
                      duration: int = 5,
                      **kwargs) -> Dict[str, Any]:
        """
        调用智谱AI视频生成接口
        
        Args:
            prompt: 视频描述提示词
            model: 使用的视频生成模型
            image_url: 可选的参考图片URL（图生视频）
            duration: 视频时长（秒）
            **kwargs: 其他参数
        
        Returns:
            API响应结果，包含生成的视频URL
        """
        if model not in FREE_VIDEO_MODELS:
            raise ValueError(f"不支持的视频生成模型: {model}. 支持的模型: {FREE_VIDEO_MODELS}")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            **kwargs
        }
        
        # 如果提供了参考图片，则添加到payload中
        if image_url:
            payload["image_url"] = image_url
        
        try:
            response = requests.post(
                ZHIPU_VIDEO_ENDPOINT,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            if response.status_code >= 400:
                raise Exception(f"视频生成API调用失败: {response.status_code} {response.text}")
            initial = response.json()
            # 异步任务：返回体含 task_status=PROCESSING，需要轮询异步结果
            if isinstance(initial, dict) and initial.get("task_status") == "PROCESSING" and (initial.get("id") or initial.get("request_id")):
                task_id = initial.get("id") or initial.get("request_id")
                return self._poll_async_result(task_id, timeout_seconds=180)
            return initial
        except requests.exceptions.RequestException as e:
            raise Exception(f"视频生成API调用失败: {str(e)}")

    def _poll_async_result(self, task_id: str, timeout_seconds: int = 180, interval_seconds: int = 3) -> Dict[str, Any]:
        """轮询异步任务结果（视频/异步对话等）"""
        import time
        from .config import ZHIPU_ASYNC_RESULT_ENDPOINT, ZHIPU_VIDEO_RESULT_ENDPOINT

        deadline = time.time() + timeout_seconds
        last_payload = None
        while time.time() < deadline:
            try:
                # 1) /videos/result?id=...
                resp = requests.get(
                    ZHIPU_VIDEO_RESULT_ENDPOINT,
                    headers=self.headers,
                    params={"id": task_id},
                    timeout=30,
                )
                if resp.status_code == 404:
                    # 2) /videos/result/{id}
                    resp = requests.get(
                        f"{ZHIPU_VIDEO_RESULT_ENDPOINT}/{task_id}",
                        headers=self.headers,
                        timeout=30,
                    )
                if resp.status_code == 404:
                    # 3) /async-result/{id}
                    resp = requests.get(
                        f"{ZHIPU_ASYNC_RESULT_ENDPOINT}/{task_id}",
                        headers=self.headers,
                        timeout=30,
                    )
                if resp.status_code == 404:
                    # 4) /async-result?request_id=...
                    resp = requests.get(
                        ZHIPU_ASYNC_RESULT_ENDPOINT,
                        headers=self.headers,
                        params={"request_id": task_id},
                        timeout=30,
                    )
                if resp.status_code >= 400:
                    raise Exception(f"查询异步结果失败: {resp.status_code} {resp.text}")
                data = resp.json()
                last_payload = data
                # 常见字段：task_status: PROCESSING | SUCCESS | FAILED
                status = data.get("task_status") or data.get("status")
                if status in ("SUCCESS", "SUCCEEDED", "DONE"):
                    return data
                if status in ("FAILED", "ERROR"):
                    raise Exception(f"异步任务失败: {data}")
                time.sleep(interval_seconds)
            except requests.exceptions.RequestException as e:
                raise Exception(f"查询异步结果API调用失败: {str(e)}")
        # 超时返回最后一次payload，便于观察
        raise Exception(f"查询异步结果超时，最后状态: {last_payload}") 