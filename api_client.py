import requests
import json
import base64
import os
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import io
import numpy as np

from .config import (ZHIPU_CHAT_ENDPOINT, ZHIPU_IMAGE_ENDPOINT, ZHIPU_VIDEO_ENDPOINT, 
                    ALL_CHAT_MODELS, ALL_IMAGE_MODELS, FREE_VIDEO_MODELS)


class ZhipuAPIClient:
    """智谱AI API客户端"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.debug_enabled = os.environ.get("ZHIPU_DEBUG", "0") in ("1", "true", "True")
    
    def _safe_float(self, value: Any, default: Optional[float] = None, min_value: Optional[float] = None, max_value: Optional[float] = None) -> Optional[float]:
        """将任意输入安全转换为 float，并裁剪范围。转换失败返回 default。"""
        try:
            if value is None:
                return default
            # 允许字符串数字，例如 "0.70"
            result = float(value)
            if min_value is not None and result < min_value:
                result = min_value
            if max_value is not None and result > max_value:
                result = max_value
            # 统一保留三位小数，避免 0.7000000000000002 之类的浮点误差
            result = round(result, 3)
            return result
        except (TypeError, ValueError):
            return default
    
    def _safe_int(self, value: Any, default: Optional[int] = None, min_value: Optional[int] = None, max_value: Optional[int] = None) -> Optional[int]:
        """将任意输入安全转换为 int，并裁剪范围。转换失败返回 default。"""
        try:
            if value is None:
                return default
            result = int(value)
            if min_value is not None and result < min_value:
                result = min_value
            if max_value is not None and result > max_value:
                result = max_value
            return result
        except (TypeError, ValueError):
            return default

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
    
    def prepare_multi_image_content(self, text: str, images: List[Union[Image.Image, np.ndarray]]) -> List[Dict[str, Any]]:
        """准备多图片消息内容"""
        content = []
        
        # 添加文本内容
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        # 添加所有图片内容
        for image in images:
            if image is not None:
                image_base64 = self.image_to_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                })
        
        return content
    
    def _is_vision_model(self, model: str) -> bool:
        m = (model or "").lower()
        return m.startswith("glm-4v") or m.startswith("glm-4.1v") or m.startswith("glm-4.5v")
    
    def _debug_print_payload(self, payload: Dict[str, Any]):
        if not self.debug_enabled:
            return
        try:
            sanitized = json.loads(json.dumps(payload))  # deep copy
            # 截断图片 base64，避免刷屏
            for msg in sanitized.get("messages", []):
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if isinstance(url, str) and url.startswith("data:image") and len(url) > 128:
                                part["image_url"]["url"] = url[:128] + "...<truncated>"
            print("[ZHIPU_DEBUG] Request payload:", json.dumps(sanitized, ensure_ascii=False))
        except Exception:
            pass
    
    def _debug_print_response(self, data: Dict[str, Any]):
        if not self.debug_enabled:
            return
        try:
            brief = {
                "has_choices": isinstance(data, dict) and bool(data.get("choices")),
                "num_choices": len(data.get("choices", [])) if isinstance(data.get("choices"), list) else 0,
                "finish_reason": (data.get("choices", [{}])[0] or {}).get("finish_reason") if data.get("choices") else None,
                "has_video_result": bool(data.get("video_result")),
                "has_web_search": bool(data.get("web_search")),
                "content_preview": ((data.get("choices", [{}])[0] or {}).get("message", {}).get("content") or "")[:120],
                "reasoning_preview": ((data.get("choices", [{}])[0] or {}).get("message", {}).get("reasoning_content") or "")[:120],
            }
            print("[ZHIPU_DEBUG] Response brief:", json.dumps(brief, ensure_ascii=False))
        except Exception:
            pass
    
    def _clean_response_text(self, text: str) -> str:
        """清理响应文本中的格式标记"""
        if not text:
            return text
        # 清理可能的格式标记
        cleaned = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '').strip()
        return cleaned
    
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
        """
        if model not in ALL_CHAT_MODELS:
            raise ValueError(f"不支持的对话模型: {model}. 支持的模型: {ALL_CHAT_MODELS}")
        
        # 强制类型与范围
        temperature_val = self._safe_float(temperature, default=0.7, min_value=0.0, max_value=1.0)
        top_p_val = self._safe_float(top_p, default=0.9, min_value=0.0, max_value=1.0)
        max_tokens_val = self._safe_int(max_tokens, default=1024, min_value=1, max_value=8192)
        
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature_val,
            "stream": stream,
        }
        # 透传 request_id（用于绕过缓存/追踪请求）
        if kwargs.get("request_id"):
            payload["request_id"] = str(kwargs.get("request_id"))
        
        # 视觉模型默认不显式传 top_p
        if not self._is_vision_model(model):
            payload["top_p"] = top_p_val
        
        # 参数兼容：部分思考类/视觉模型要求使用 max_new_tokens
        if isinstance(max_tokens_val, int) and max_tokens_val > 0:
            if "thinking" in model or model.endswith("-thinking-flash") or model.startswith("glm-4.1v") or model.startswith("glm-4.5v"):
                payload["max_new_tokens"] = max_tokens_val
            else:
                payload["max_tokens"] = max_tokens_val
        
        self._debug_print_payload(payload)
        
        try:
            response = requests.post(
                ZHIPU_CHAT_ENDPOINT,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            if response.status_code >= 400:
                # 失败时强制输出一次精简调试信息
                if not self.debug_enabled:
                    try:
                        brief = {
                            "model": payload.get("model"),
                            "temperature": payload.get("temperature"),
                            "top_p": payload.get("top_p", "<omitted>"),
                            "max_tokens": payload.get("max_tokens"),
                            "max_new_tokens": payload.get("max_new_tokens"),
                            "stream": payload.get("stream"),
                            "request_id": payload.get("request_id"),
                            "messages_len": len(payload.get("messages", [])),
                        }
                        print("[ZHIPU_ERROR] Brief payload:", json.dumps(brief, ensure_ascii=False))
                    except Exception:
                        pass
                else:
                    self._debug_print_payload(payload)
                raise Exception(f"API调用失败: {response.status_code} {response.text}")
            data = response.json()
            self._debug_print_response(data)
            return data
        except requests.exceptions.RequestException as e:
            raise Exception(f"API调用失败: {str(e)}")
    
    def simple_chat(self, 
                   prompt: str, 
                   image: Optional[Union[Image.Image, np.ndarray]] = None,
                   model: str = "glm-4.5",
                   temperature: float = 0.7,
                   max_tokens: int = 1024,
                   system_prompt: str = "",
                   **kwargs) -> str:
        """
        简单的对话接口
        """
        # 准备消息内容（用户消息，可能包含文本+图片）
        content = self.prepare_message_content(prompt, image)
        
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        
        # 统一做数值校验
        temperature_val = self._safe_float(temperature, default=0.7, min_value=0.0, max_value=1.0)
        max_tokens_val = self._safe_int(max_tokens, default=1024, min_value=1, max_value=8192)
        
        # 调用API
        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature_val,
            max_tokens=max_tokens_val,
            request_id=kwargs.get("request_id"),
        )
        
        # 提取回复内容（兼容 reasoning_content / audio 等）
        if isinstance(response, dict) and "choices" in response and len(response.get("choices", [])) > 0:
            msg = (response["choices"][0] or {}).get("message", {})
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("reasoning_content") or "").strip()
            if not text and isinstance(msg.get("audio"), dict):
                audio_id = msg["audio"].get("id")
                if audio_id:
                    text = f"[音频返回] audio_id={audio_id}"
            if text:
                return self._clean_response_text(text)
            # 调试：如果仍然拿不到文本，打印一份简要信息
            if self.debug_enabled:
                try:
                    print("[ZHIPU_DEBUG] Empty content, raw message:", json.dumps(msg, ensure_ascii=False)[:500])
                except Exception:
                    pass
            raise Exception(f"API未返回文本内容: {response}")
        else:
            raise Exception(f"API返回格式异常: {response}")
    
    def multi_image_chat(self, 
                        prompt: str, 
                        images: List[Union[Image.Image, np.ndarray]],
                        model: str = "glm-4v-flash",
                        temperature: float = 0.7,
                        max_tokens: int = 1024,
                        system_prompt: str = "",
                        **kwargs) -> str:
        """
        多图片对话接口
        """
        # 准备多图片消息内容
        content = self.prepare_multi_image_content(prompt, images)
        
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": content})
        
        # 统一做数值校验
        temperature_val = self._safe_float(temperature, default=0.7, min_value=0.0, max_value=1.0)
        max_tokens_val = self._safe_int(max_tokens, default=1024, min_value=1, max_value=8192)
        
        # 调用API
        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature_val,
            max_tokens=max_tokens_val,
            request_id=kwargs.get("request_id"),
        )
        
        # 提取回复内容（兼容 reasoning_content / audio 等）
        if isinstance(response, dict) and "choices" in response and len(response.get("choices", [])) > 0:
            msg = (response["choices"][0] or {}).get("message", {})
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("reasoning_content") or "").strip()
            if not text and isinstance(msg.get("audio"), dict):
                audio_id = msg["audio"].get("id")
                if audio_id:
                    text = f"[音频返回] audio_id={audio_id}"
            if text:
                return self._clean_response_text(text)
            # 调试：如果仍然拿不到文本，打印一份简要信息
            if self.debug_enabled:
                try:
                    print("[ZHIPU_DEBUG] Empty content, raw message:", json.dumps(msg, ensure_ascii=False)[:500])
                except Exception:
                    pass
            raise Exception(f"API未返回文本内容: {response}")
        else:
            raise Exception(f"API返回格式异常: {response}")
    
    def generate_image(self,
                      prompt: str,
                      model: str = "glm-image",
                      size: str = "1280x1280",
                      n: int = 1,
                      quality: str = "standard",
                      watermark_enabled: bool = True,
                      **kwargs) -> Dict[str, Any]:
        """
        调用智谱AI图片生成接口
        
        Args:
            prompt: 图片描述提示词
            model: 使用的图片生成模型
                - cogview-3-flash: CogView-3快速版（免费）
                - glm-image: GLM-Image新旗舰图像生成模型（0.1元/次）
            size: 图片尺寸
                - CogView-3: 1024x1024, 768x1344, 1344x768, 1536x1024, 1024x1536
                - GLM-Image: 1280x1280, 1568x1056, 1056x1568, 1472x1088, 1088x1472, 1728x960, 960x1728
                  或自定义尺寸（512-2048，32的倍数）
            n: 生成图片数量（仅CogView系列支持）
            quality: 图片质量 (standard, hd)（仅CogView系列支持）
            watermark_enabled: 是否添加水印（默认True）。False需在平台签署免责声明。
            **kwargs: 其他参数
        
        Returns:
            API响应结果，包含生成的图片URL
        """
        if model not in ALL_IMAGE_MODELS:
            raise ValueError(f"不支持的图片生成模型: {model}. 支持的模型: {ALL_IMAGE_MODELS}")
        
        # 构建请求payload
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "watermark_enabled": watermark_enabled,
        }
        
        # GLM-Image 只支持 prompt 和 size，CogView 支持更多参数
        if model != "glm-image":
            payload["n"] = n
            payload["quality"] = quality
            # 添加其他可选参数
            payload.update(kwargs)
        
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