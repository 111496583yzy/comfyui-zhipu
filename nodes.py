import os
import torch
import requests
import io
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional

from .api_client import ZhipuAPIClient
from .config import (ALL_CHAT_MODELS, VISION_CHAT_MODELS, ALL_IMAGE_MODELS,
                    FREE_VIDEO_MODELS, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_TOP_P)
from .local_config import load_api_key, save_api_key


class ZhipuAPIConfig:
    """智谱AI API配置节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "请输入智谱AI API Key（留空将从插件目录 config.json 读取）"
                }),
                "remember": ("BOOLEAN", {
                    "default": True,
                }),
            }
        }

    RETURN_TYPES = ("ZHIPU_CLIENT",)
    RETURN_NAMES = ("zhipu_client",)
    FUNCTION = "create_client"
    CATEGORY = "ZhipuAI"

    def create_client(self, api_key: str, remember: bool = True) -> Tuple[ZhipuAPIClient]:
        """创建智谱AI客户端
        - 若 api_key 非空且 remember=True，则保存到插件目录 config.json
        - 若 api_key 为空，则尝试从插件目录 config.json 读取
        """
        key = (api_key or "").strip()
        if not key:
            # 仅从插件目录配置读取
            key = (load_api_key() or "").strip()
        if not key:
            raise ValueError("API Key不能为空（可在此输入一次并勾选记住，或在插件目录 config.json 中设置）")

        # 如果是新输入的且需要记住，则保存到插件目录
        if api_key.strip() and remember:
            try:
                save_api_key(key)
            except Exception as e:
                print(f"[ZhipuAPIConfig] 保存API Key失败: {e}")
        
        client = ZhipuAPIClient(key)
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2_147_483_647,
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
                     seed: int = 0,
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
            
            # 调用API（将 seed 写入 request_id 用于去缓存）
            # 确保 request_id 长度至少为 6 个字符（智谱AI要求）
            request_id = f"txt-{str(seed).zfill(4)}" if seed else None
            response = zhipu_client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                request_id=request_id,
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2_147_483_647,
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
                           seed: int = 0,
                           system_prompt: str = "") -> Tuple[str]:
        """生成基于图片的文本回复"""
        try:
            # 转换图片格式（从ComfyUI的tensor格式转换）
            images_list = []
            if isinstance(image, torch.Tensor):
                # ComfyUI图片格式通常是 [batch, height, width, channels]
                if image.dim() == 4:
                    # 处理批次中的所有图片
                    batch_size = image.shape[0]
                    for i in range(batch_size):
                        img_np = image[i].cpu().numpy()
                        images_list.append(img_np)
                else:
                    # 单张图片
                    images_list.append(image.cpu().numpy())
            else:
                images_list.append(image)
            
            # 若未填写文本提示，提供一个默认提示，避免只有图片导致400
            safe_prompt = prompt.strip() or "请描述这些图片的关键信息与要点。"
            
            # 调用多图片对话接口（将 seed 写入 request_id 用于去缓存）
            # 确保 request_id 长度至少为 6 个字符（智谱AI要求）
            request_id = f"vis-{str(seed).zfill(4)}" if seed else None
            reply = zhipu_client.multi_image_chat(
                prompt=safe_prompt,
                images=images_list,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                request_id=request_id,
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
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2_147_483_647,
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
                     seed: int = 0,
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
            
            # 调用API（将 seed 写入 request_id 用于去缓存）
            # 确保 request_id 长度至少为 6 个字符（智谱AI要求）
            request_id = f"hist-{str(seed).zfill(4)}" if seed else None
            response = zhipu_client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                request_id=request_id,
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
    """智谱AI图片生成节点
    
    支持模型：
    - cogview-3-flash: CogView-3快速版（免费）
    - glm-image: GLM-Image新旗舰图像生成模型（0.1元/次）
      - 推荐尺寸: 1280x1280, 1568x1056, 1056x1568, 1472x1088, 1088x1472, 1728x960, 960x1728
      - 自定义尺寸: 长宽需在512px-2048px范围内，且长宽均需为32的整数倍
    """
    
    # CogView-3 支持的尺寸
    COGVIEW_SIZES = [
        "1024x1024",  # 1:1
        "768x1344",   # 9:16 竖版
        "1344x768",   # 16:9 横版
        "1536x1024",  # 3:2
        "1024x1536",  # 2:3
    ]
    
    # GLM-Image 推荐尺寸
    GLM_IMAGE_SIZES = [
        "1280x1280",  # 1:1
        "1568x1056",  # 约 3:2
        "1056x1568",  # 约 2:3
        "1472x1088",  # 约 4:3
        "1088x1472",  # 约 3:4
        "1728x960",   # 16:9
        "960x1728",   # 9:16
    ]
    
    # 合并所有预设尺寸
    ALL_PRESET_SIZES = list(dict.fromkeys(COGVIEW_SIZES + GLM_IMAGE_SIZES))  # 去重保持顺序
    
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
                "model": (ALL_IMAGE_MODELS, {
                    "default": "glm-image"
                }),
                "size": (["自定义"] + cls.ALL_PRESET_SIZES, {
                    "default": "1280x1280"
                }),
            },
            "optional": {
                "custom_width": ("INT", {
                    "default": 1280,
                    "min": 512,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "自定义宽度（仅在size选择'自定义'时生效），需为32的整数倍"
                }),
                "custom_height": ("INT", {
                    "default": 1280,
                    "min": 512,
                    "max": 2048,
                    "step": 32,
                    "tooltip": "自定义高度（仅在size选择'自定义'时生效），需为32的整数倍"
                }),
                "quality": (["standard", "hd"], {
                    "default": "standard",
                    "tooltip": "图片质量（仅CogView系列支持）"
                }),
                "n": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                    "step": 1,
                    "tooltip": "生成图片数量（仅CogView系列支持）"
                }),
                "disable_watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否去水印（需在智谱AI开放平台签署免责声明后生效）"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_urls", "response_info")
    FUNCTION = "generate_image"
    CATEGORY = "ZhipuAI"

    def generate_image(self, 
                      zhipu_client: ZhipuAPIClient, 
                      prompt: str,
                      model: str = "glm-image",
                      size: str = "1280x1280",
                      custom_width: int = 1280,
                      custom_height: int = 1280,
                      quality: str = "standard",
                      n: int = 1,
                      disable_watermark: bool = False) -> Tuple[torch.Tensor, str, str]:
        """生成图片"""
        try:
            # 处理自定义尺寸
            if size == "自定义":
                # 确保尺寸是32的倍数
                custom_width = (custom_width // 32) * 32
                custom_height = (custom_height // 32) * 32
                # 确保在范围内
                custom_width = max(512, min(2048, custom_width))
                custom_height = max(512, min(2048, custom_height))
                actual_size = f"{custom_width}x{custom_height}"
            else:
                actual_size = size
            
            # 准备参数
            kwargs = {
                "watermark_enabled": not disable_watermark
            }
            
            # 根据模型类型调用不同的参数
            if model == "glm-image":
                # GLM-Image 只支持 prompt 和 size 以及 watermark_enabled
                response = zhipu_client.generate_image(
                    prompt=prompt,
                    model=model,
                    size=actual_size,
                    **kwargs
                )
            else:
                # CogView 系列支持更多参数
                response = zhipu_client.generate_image(
                    prompt=prompt,
                    model=model,
                    size=actual_size,
                    quality=quality,
                    n=n,
                    **kwargs
                )
            
            # 提取图片URLs
            if "data" in response and len(response["data"]) > 0:
                image_urls = []
                for item in response["data"]:
                    if "url" in item:
                        image_urls.append(item["url"])
                
                urls_text = "\n".join(image_urls)
                info_text = f"成功生成 {len(image_urls)} 张图片，模型: {model}, 尺寸: {actual_size}"
                
                # 下载并转换为ComfyUI格式 (Batch, Height, Width, Channel)
                output_images = []
                for url in image_urls:
                    try:
                        resp = requests.get(url, timeout=60)
                        resp.raise_for_status()
                        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        img_np = np.array(img).astype(np.float32) / 255.0
                        img_tensor = torch.from_numpy(img_np)[None,] # Add batch dimension
                        output_images.append(img_tensor)
                    except Exception as e:
                        print(f"下载图片失败 {url}: {e}")
                
                if output_images:
                    final_image = torch.cat(output_images, dim=0)
                    return (final_image, urls_text, info_text)
                else:
                    # 下载失败返回空黑色图片
                    empty = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
                    return (empty, urls_text, info_text + " (图片下载失败)")
                    
            else:
                raise Exception(f"图片生成API返回格式异常: {response}")
            
        except Exception as e:
            error_msg = f"图片生成失败: {str(e)}"
            print(error_msg)
            # 失败返回空图片和错误信息
            empty = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (empty, "", error_msg)


class ZhipuVideoGeneration:
    """智谱AI视频生成节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zhipu_client": ("ZHIPU_CLIENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一只小猫在花园里玩耍",
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
                    "multiline": False,
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
                       image_url: Optional[str] = None,
                       duration: int = 5) -> Tuple[str, str]:
        """生成视频（异步）"""
        try:
            response = zhipu_client.generate_video(
                prompt=prompt,
                model=model,
                image_url=image_url if image_url else None,
                duration=duration
            )
            
            # 解析视频URL
            url = response.get("url") or response.get("video_url") or ""
            if not url and isinstance(response, dict):
                # 兼容异步结果结构
                data = response.get("data") or response
                url = (data.get("url") if isinstance(data, dict) else "") or url
            
            info_text = f"视频生成任务状态: {response.get('task_status') or response.get('status') or 'UNKNOWN'}"
            return (url or "", info_text)
        except Exception as e:
            error_msg = f"视频生成失败: {str(e)}"
            print(error_msg)
            return ("", error_msg)

# 节点注册
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