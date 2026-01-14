# 智谱AI API 配置
ZHIPU_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_CHAT_ENDPOINT = f"{ZHIPU_API_BASE_URL}/chat/completions"
ZHIPU_IMAGE_ENDPOINT = f"{ZHIPU_API_BASE_URL}/images/generations"
ZHIPU_VIDEO_ENDPOINT = f"{ZHIPU_API_BASE_URL}/videos/generations"
ZHIPU_ASYNC_RESULT_ENDPOINT = f"{ZHIPU_API_BASE_URL}/async-result"
ZHIPU_VIDEO_RESULT_ENDPOINT = f"{ZHIPU_API_BASE_URL}/videos/result"

# 免费对话模型列表
FREE_CHAT_MODELS = [
    "glm-4.5-flash",           # GLM-4.5快速版（免费）
    "glm-4.1v-thinking-flash", # GLM-4.1V思考版（免费，支持视觉）
    "glm-4-flash-250414",      # GLM-4特定版本（免费）
    "glm-4v-flash",            # GLM-4V视觉快速版（免费，支持视觉）
    "glm-z1-flash"             # GLM-Z1快速版（免费）
]

# 免费图片生成模型
FREE_IMAGE_MODELS = [
    "cogview-3-flash"
]

# 付费图片生成模型
PAID_IMAGE_MODELS = [
    "glm-image"      # GLM-Image 新旗舰图像生成模型（0.1元/次）
]

# 所有图片生成模型
ALL_IMAGE_MODELS = FREE_IMAGE_MODELS + PAID_IMAGE_MODELS

# 免费视频生成模型
FREE_VIDEO_MODELS = [
    "cogvideox-flash"
]

# 付费对话模型（根据官方文档更新）
PAID_CHAT_MODELS = [
    # GLM-4.6系列（最新旗舰模型）
    "glm-4.6",                    # GLM-4.6旗舰模型（200K上下文，128K输出）
    
    # GLM-4.5系列
    "glm-4.5",                    # GLM-4.5标准版（128K上下文，96K输出）
    "glm-4.5-x",                  # GLM-4.5极速版（128K上下文，96K输出）
    "glm-4.5-air",                # GLM-4.5高性价比版（128K上下文，96K输出）
    "glm-4.5-airx",               # GLM-4.5高性价比极速版（128K上下文，96K输出）
    "glm-4.5v",                   # GLM-4.5V视觉模型（支持视觉理解）
    
    # GLM-4系列
    "glm-4-plus",                 # GLM-4增强版（128K上下文，4K输出）
    "glm-4-air-250414",           # GLM-4高性价比版（128K上下文，16K输出）
    "glm-4-long",                # GLM-4长文本版（1M上下文，4K输出）
    "glm-4-airx",                # GLM-4极速版（8K上下文，4K输出）
    "glm-4-flashx-250414",       # GLM-4高速低价版（128K上下文，16K输出）
    "glm-4",                     # GLM-4基础模型
    "glm-4-0520",                # GLM-4特定版本（即将弃用）
    
    # GLM-Z1系列
    "glm-z1-air",                # GLM-Z1高性价比版（128K上下文，32K输出）
    "glm-z1-airx",               # GLM-Z1极速版（32K上下文，30K输出）
    "glm-z1-flashx",             # GLM-Z1高速低价版（128K上下文，32K输出）
    
    # 视觉模型
    "glm-4v",                    # GLM-4V标准版（支持视觉）
    "glm-4v-plus",               # GLM-4V增强版（支持视觉）
    "glm-4v-plus-0111",          # GLM-4V增强版0111（支持视觉）
    "glm-4.1v-thinking-flashx", # GLM-4.1V思考版增强变体（支持视觉）
]

# 所有对话模型（免费+付费）
ALL_CHAT_MODELS = FREE_CHAT_MODELS + PAID_CHAT_MODELS

# 视觉对话模型（支持图片理解的模型）
VISION_CHAT_MODELS = [
    # 免费视觉模型
    "glm-4.1v-thinking-flash",  # GLM-4.1V思考版（免费，支持视觉）
    "glm-4v-flash",             # GLM-4V快速版（免费，支持视觉）
    
    # 付费视觉模型
    "glm-4.5v",                 # GLM-4.5V视觉模型（支持视觉理解）
    "glm-4v",                   # GLM-4V标准版（支持视觉）
    "glm-4v-plus",              # GLM-4V增强版（支持视觉）
    "glm-4v-plus-0111",         # GLM-4V增强版0111（支持视觉）
    "glm-4.1v-thinking-flashx", # GLM-4.1V思考版增强变体（支持视觉）
]

# 默认参数
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOP_P = 0.9 