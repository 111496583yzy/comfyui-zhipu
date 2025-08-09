import os
import torch
from typing import Tuple, Dict, Any, Optional

from .api_client import ZhipuAPIClient
from .config import (ALL_CHAT_MODELS, VISION_CHAT_MODELS, FREE_IMAGE_MODELS, 
                    FREE_VIDEO_MODELS, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P)


class ZhipuAPIConfig:
    """智谱AI API配置节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "请输入智谱AI API Key"
                }),
            }
        }

    RETURN_TYPES = ("ZHIPU_CLIENT",)
    RETURN_NAMES = ("zhipu_client",)
    FUNCTION = "create_client"
    CATEGORY = "ZhipuAI"

    def create_client(self, api_key: str) -> Tuple[ZhipuAPIClient]:
        """创建智谱AI客户端"""
        if not api_key.strip():
            raise ValueError("API Key不能为空")
        
        client = ZhipuAPIClient(api_key)
        return (client,)


class ZhipuTextChat:
    """智谱AI文本对话节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhipu_client": ("ZHIPU_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "你好！",
                    "placeholder": "请输入您的问题或对话内容"
                }),
                "model": (ALL_CHAT_MODELS, {
                    "default": "glm-4.5-flash"
                }),
                "temperature": ("FLOAT", {
                    "default": DEFAULT_TEMPERATURE,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_tokens": ("INT", {
                    "default": DEFAULT_MAX_TOKENS,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "可选：系统提示词"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "ZhipuAI"

    def generate_text(self, 
                     zhipu_client: ZhipuAPIClient, 
                     prompt: str, 
                     model: str = "glm-4.5",
                     temperature: float = DEFAULT_TEMPERATURE,
                     max_tokens: int = DEFAULT_MAX_TOKENS,
                     system_prompt: str = "") -> Tuple[str]:
        """生成文本回复"""
        try:
            messages = []
            
            # 添加系统提示词（如果有）
            if system_prompt.strip():
                messages.append({
                    "role": "system",
                    "content": system_prompt.strip()
                })
            
            # 添加用户消息
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # 调用API
            response = zhipu_client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取回复内容
            if "choices" in response and len(response["choices"]) > 0:
                reply = response["choices"][0]["message"]["content"]
                return (reply,)
            else:
                raise Exception(f"API返回格式异常: {response}")
                
        except Exception as e:
            error_msg = f"智谱AI调用失败: {str(e)}"
            print(error_msg)
            return (error_msg,)


class ZhipuVisionChat:
    """智谱AI视觉对话节点（支持图片输入）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhipu_client": ("ZHIPU_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "请描述一下这张图片",
                    "placeholder": "请输入关于图片的问题"
                }),
                "image": ("IMAGE",),
                "model": (VISION_CHAT_MODELS, {
                    "default": "glm-4v-flash"
                }),
                "temperature": ("FLOAT", {
                    "default": DEFAULT_TEMPERATURE,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_tokens": ("INT", {
                    "default": DEFAULT_MAX_TOKENS,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "可选：系统提示词"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_vision_text"
    CATEGORY = "ZhipuAI"

    def generate_vision_text(self, 
                           zhipu_client: ZhipuAPIClient,
                           prompt: str,
                           image,
                           model: str = "glm-4v",
                           temperature: float = DEFAULT_TEMPERATURE,
                           max_tokens: int = DEFAULT_MAX_TOKENS,
                           system_prompt: str = "") -> Tuple[str]:
        """生成基于图片的文本回复"""
        try:
            # 转换图片格式（从ComfyUI的tensor格式转换）
            if isinstance(image, torch.Tensor):
                # ComfyUI图片格式通常是 [batch, height, width, channels]
                if image.dim() == 4:
                    image = image[0]  # 取第一张图片
                # 转换为numpy数组
                image_np = image.cpu().numpy()
            else:
                image_np = image
            
            # 若未填写文本提示，提供一个默认提示，避免只有图片导致400
            safe_prompt = prompt.strip() or "请描述这张图片的关键信息与要点。"
            
            # 调用简单对话接口
            reply = zhipu_client.simple_chat(
                prompt=safe_prompt,
                image=image_np,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            
            return (reply,)
            
        except Exception as e:
            error_msg = f"智谱AI视觉对话失败: {str(e)}"
            print(error_msg)
            return (error_msg,)


class ZhipuChatHistory:
    """智谱AI对话历史节点"""
    
    def __init__(self):
        self.history = []
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhipu_client": ("ZHIPU_CLIENT",),
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入对话内容"
                }),
                "model": (ALL_CHAT_MODELS, {
                    "default": "glm-4.5-flash"
                }),
                "temperature": ("FLOAT", {
                    "default": DEFAULT_TEMPERATURE,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_tokens": ("INT", {
                    "default": DEFAULT_MAX_TOKENS,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "可选：系统提示词"
                }),
                "image": ("IMAGE",),
                "reset_history": ("BOOLEAN", {
                    "default": False
                }),
                "chat_history": ("ZHIPU_HISTORY",),
            }
        }

    RETURN_TYPES = ("STRING", "ZHIPU_HISTORY")
    RETURN_NAMES = ("response", "chat_history")
    FUNCTION = "continue_chat"
    CATEGORY = "ZhipuAI"

    def continue_chat(self, 
                     zhipu_client: ZhipuAPIClient,
                     user_input: str,
                     model: str = "glm-4.5",
                     temperature: float = DEFAULT_TEMPERATURE,
                     max_tokens: int = DEFAULT_MAX_TOKENS,
                     system_prompt: str = "",
                     image=None,
                     reset_history: bool = False,
                     chat_history=None) -> Tuple[str, list]: 
        """继续对话并维护历史记录"""
        try:
            # 如果需要重置历史或者没有传入历史，则初始化
            if reset_history or chat_history is None:
                history = []
            else:
                history = chat_history.copy() if isinstance(chat_history, list) else []
            
            # 构建消息列表
            messages = []
            
            # 确保系统提示词进入历史（只写入一次，随后每轮自动带上）
            if system_prompt.strip() and not history:
                history.append({
                    "role": "system",
                    "content": system_prompt.strip()
                })
            
            # 添加历史对话
            messages.extend(history)
            
            # 添加当前用户输入（支持图片 + 文本）
            if image is not None:
                # 将tensor图片转为numpy交给client处理
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:
                        image = image[0]
                    image_np = image.cpu().numpy()
                else:
                    image_np = image
                content = zhipu_client.prepare_message_content(user_input, image_np)
            else:
                content = user_input

            messages.append({
                "role": "user",
                "content": content
            })
            
            # 调用API
            response = zhipu_client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # 提取回复内容
            if "choices" in response and len(response["choices"]) > 0:
                reply = response["choices"][0]["message"]["content"]
                
                # 更新历史记录（仅记录文本，避免把图片base64写入历史导致上下文过大）
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": reply})
                
                return (reply, history)
            else:
                raise Exception(f"API返回格式异常: {response}")
                
        except Exception as e:
            error_msg = f"智谱AI对话失败: {str(e)}"
            print(error_msg)
            return (error_msg, chat_history if chat_history else [])


class ZhipuImageGeneration:
    """智谱AI图片生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhipu_client": ("ZHIPU_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只可爱的熊猫在竹林中",
                    "placeholder": "请输入图片描述"
                }),
                "model": (FREE_IMAGE_MODELS, {
                    "default": "cogview-3-flash"
                }),
                "size": (["1024x1024", "768x1344", "1344x768", "1536x1024", "1024x1536"], {
                    "default": "1024x1024"
                }),
                "quality": (["standard", "hd"], {
                    "default": "standard"
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_urls", "response_info")
    FUNCTION = "generate_image"
    CATEGORY = "ZhipuAI"

    def generate_image(self, 
                      zhipu_client: ZhipuAPIClient, 
                      prompt: str,
                      model: str = "cogview-3-flash",
                      size: str = "1024x1024",
                      quality: str = "standard",
                      n: int = 1) -> Tuple[str, str]:
        """生成图片"""
        try:
            response = zhipu_client.generate_image(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                n=n
            )
            
            # 提取图片URLs
            if "data" in response and len(response["data"]) > 0:
                image_urls = []
                for item in response["data"]:
                    if "url" in item:
                        image_urls.append(item["url"])
                
                urls_text = "\n".join(image_urls)
                info_text = f"成功生成 {len(image_urls)} 张图片，模型: {model}"
                
                return (urls_text, info_text)
            else:
                raise Exception(f"API返回格式异常: {response}")
                
        except Exception as e:
            error_msg = f"智谱AI图片生成失败: {str(e)}"
            print(error_msg)
            return ("", error_msg)


class ZhipuVideoGeneration:
    """智谱AI视频生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhipu_client": ("ZHIPU_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只猫咪在花园里玩耍",
                    "placeholder": "请输入视频描述"
                }),
                "model": (FREE_VIDEO_MODELS, {
                    "default": "cogvideox-flash"
                }),
                "duration": ("INT", {
                    "default": 5,
                    "min": 2,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                "image_url": ("STRING", {
                    "default": "",
                    "placeholder": "可选：参考图片URL（图生视频）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_url", "response_info")
    FUNCTION = "generate_video"
    CATEGORY = "ZhipuAI"

    def generate_video(self, 
                      zhipu_client: ZhipuAPIClient, 
                      prompt: str,
                      model: str = "cogvideox-flash",
                      duration: int = 5,
                      image_url: str = "") -> Tuple[str, str]:
        """生成视频"""
        try:
            # 准备参数
            kwargs = {}
            if image_url.strip():
                kwargs["image_url"] = image_url.strip()
            
            response = zhipu_client.generate_video(
                prompt=prompt,
                model=model,
                duration=duration,
                **kwargs
            )
            
            # 处理异步/同步两种格式
            # 异步查询返回可能包含 task_status 与 url 字段
            if "video_result" in response and response.get("video_result"):
                video_url = response["video_result"][0].get("url", "")
                info_text = f"成功生成视频，时长: {duration}秒，模型: {model}"
                return (video_url, info_text)
            if "data" in response and response.get("data"):
                item = response["data"][0]
                video_url = item.get("url", "") or item.get("video_url", "")
                if video_url:
                    info_text = f"成功生成视频，时长: {duration}秒，模型: {model}"
                    return (video_url, info_text)
            # 有些异步查询成功后直接在根字段给 url
            if response.get("url"):
                return (response.get("url"), f"成功生成视频，时长: {duration}秒，模型: {model}")
            # 仍未拿到URL
            raise Exception(f"API返回格式异常: {response}")
                
        except Exception as e:
            error_msg = f"智谱AI视频生成失败: {str(e)}"
            print(error_msg)
            return ("", error_msg)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ZhipuAPIConfig": ZhipuAPIConfig,
    "ZhipuTextChat": ZhipuTextChat,
    "ZhipuVisionChat": ZhipuVisionChat,
    "ZhipuChatHistory": ZhipuChatHistory,
    "ZhipuImageGeneration": ZhipuImageGeneration,
    "ZhipuVideoGeneration": ZhipuVideoGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZhipuAPIConfig": "智谱AI API配置",
    "ZhipuTextChat": "智谱AI文本对话", 
    "ZhipuVisionChat": "智谱AI视觉对话",
    "ZhipuChatHistory": "智谱AI对话历史",
    "ZhipuImageGeneration": "智谱AI图片生成",
    "ZhipuVideoGeneration": "智谱AI视频生成",
} 