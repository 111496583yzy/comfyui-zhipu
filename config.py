# 智谱AI API 配置
ZHIPU_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_CHAT_ENDPOINT = f"{ZHIPU_API_BASE_URL}/chat/completions"
ZHIPU_IMAGE_ENDPOINT = f"{ZHIPU_API_BASE_URL}/images/generations"
ZHIPU_VIDEO_ENDPOINT = f"{ZHIPU_API_BASE_URL}/videos/generations"
ZHIPU_ASYNC_RESULT_ENDPOINT = f"{ZHIPU_API_BASE_URL}/async-result"
ZHIPU_VIDEO_RESULT_ENDPOINT = f"{ZHIPU_API_BASE_URL}/videos/result"

# 免费对话模型列表
FREE_CHAT_MODELS = [
    "glm-4.5-flash",
    "glm-4.1v-thinking-flash",
    "glm-4-flash-250414",
    "glm-4v-flash",
    "glm-z1-flash"
]

# 免费图片生成模型
FREE_IMAGE_MODELS = [
    "cogview-3-flash"
]

# 免费视频生成模型
FREE_VIDEO_MODELS = [
    "cogvideox-flash"
]

# 传统付费模型（保留兼容性）
PAID_CHAT_MODELS = [
    "glm-4.5",
    "glm-4",
    "glm-4-0520", 
    "glm-4-plus",
    "glm-4-air",
    "glm-4-airx",
    "glm-4-long",
    "glm-4-flashx",
    "glm-4v",
    "glm-4v-plus",
    # 新增可选视觉与版本化模型
    "glm-4.5v",
    "glm-4v-plus-0111",
    "glm-4.1v-thinking-flashx",
]

# 所有对话模型（免费+付费）
ALL_CHAT_MODELS = FREE_CHAT_MODELS + PAID_CHAT_MODELS

# 视觉对话模型（支持图片理解的模型）
VISION_CHAT_MODELS = [
    # 免费视觉模型
    "glm-4.1v-thinking-flash",  # GLM-4.1V思考版（支持视觉）
    "glm-4v-flash",             # GLM-4V快速版
    # 付费视觉模型
    "glm-4v",                   # GLM-4V标准版
    "glm-4v-plus",              # GLM-4V增强版
    # 新增视觉模型变体
    "glm-4.5v",
    "glm-4v-plus-0111",
    "glm-4.1v-thinking-flashx",
]

# 默认参数
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 0.9 