import os
import io
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image

try:
    import cv2  # opencv-python
except Exception as _e:
    cv2 = None


class QwenBlockwiseCanny:
    """使用 DiffSynth-Studio 的 Qwen-Image Blockwise ControlNet Canny 的自定义节点

    依赖：
    - DiffSynth-Studio 源码安装（提供 `diffsynth` 包）
      git clone https://github.com/modelscope/DiffSynth-Studio.git && cd DiffSynth-Studio && pip install -e .
    - modelscope、opencv-python、torch
    """

    _PIPELINE = None
    _DEVICE = None
    _DTYPE = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "请输入提示词"
                }),
                "low_threshold": ("INT", {"default": 100, "min": 0, "max": 255, "step": 1}),
                "high_threshold": ("INT", {"default": 200, "min": 0, "max": 255, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2_147_483_647, "step": 1}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            },
            "optional": {
                "model_cache_dir": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "可选：模型缓存目录（留空自动）"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "Qwen-Image"

    def _ensure_pipeline(self, device: str, dtype: str, model_cache_dir: Optional[str] = None):
        if self._PIPELINE is not None and self._DEVICE == device and self._DTYPE == dtype:
            return self._PIPELINE
        try:
            from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
        except Exception as e:
            raise RuntimeError(
                "未找到 DiffSynth-Studio，请先安装源码：\n"
                "git clone https://github.com/modelscope/DiffSynth-Studio.git\n"
                "cd DiffSynth-Studio && pip install -e .\n"
                f"导入错误：{e}"
            )

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        kwargs = {
            "torch_dtype": torch_dtype,
            "device": device,
            "model_configs": [
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
                ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Blockwise-ControlNet-Canny", origin_file_pattern="model.safetensors"),
            ],
            "tokenizer_config": ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        }
        if model_cache_dir and model_cache_dir.strip():
            os.environ["MODELSCOPE_CACHE"] = model_cache_dir.strip()

        pipe = QwenImagePipeline.from_pretrained(**kwargs)
        self._PIPELINE = pipe
        self._DEVICE = device
        self._DTYPE = dtype
        return pipe

    def _tensor_to_pil(self, image_tensor) -> Image.Image:
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            arr = image_tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(image_tensor)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        arr = arr[:, :, :3]
        return Image.fromarray(arr)

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        arr = arr.astype(np.float32) / 255.0
        h, w, c = arr.shape
        return torch.from_numpy(arr).view(1, h, w, c)

    def _to_canny(self, img: Image.Image, low: int, high: int) -> Image.Image:
        if cv2 is None:
            raise RuntimeError("未安装 opencv-python，请先安装：pip install opencv-python")
        gray = np.array(img.convert("L"))
        edges = cv2.Canny(gray, threshold1=int(low), threshold2=int(high))
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    def generate(self,
                 image,
                 prompt: str,
                 low_threshold: int = 100,
                 high_threshold: int = 200,
                 seed: int = 0,
                 device: str = "cuda",
                 dtype: str = "bfloat16",
                 model_cache_dir: str = "") -> Tuple[torch.Tensor]:
        base_pil = self._tensor_to_pil(image)
        canny_pil = self._to_canny(base_pil, low_threshold, high_threshold)

        pipe = self._ensure_pipeline(device=device, dtype=dtype, model_cache_dir=model_cache_dir)

        try:
            from diffsynth.pipelines.qwen_image import ControlNetInput
        except Exception:
            raise RuntimeError("DiffSynth-Studio 版本不匹配：缺少 ControlNetInput。请更新至仓库主分支后重试。")

        # Qwen-Image 常用分辨率为 1328x1328，这里按需缩放
        target = 1328
        canny_resized = canny_pil.resize((target, target))

        out_img = pipe(
            prompt or "",
            seed=int(seed) if seed else 0,
            blockwise_controlnet_inputs=[ControlNetInput(image=canny_resized)],
        )
        if isinstance(out_img, Image.Image):
            return (self._pil_to_tensor(out_img),)
        # 兼容 pipeline 可能返回列表的情况
        if isinstance(out_img, (list, tuple)) and len(out_img) > 0 and isinstance(out_img[0], Image.Image):
            return (self._pil_to_tensor(out_img[0]),)
        raise RuntimeError(f"未知的输出类型：{type(out_img)}") 