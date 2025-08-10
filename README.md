# 智谱AI ComfyUI 插件

这是一个用于在ComfyUI中集成智谱AI大语言模型的插件，支持文本对话、视觉对话和多轮对话功能。

## 功能特性

- ✅ **文本对话**：支持智谱AI的免费和付费对话模型（GLM-4系列）
- ✅ **视觉对话**：支持图片理解和多模态对话（GLM-4V系列）
- ✅ **多轮对话**：维护对话历史，支持连续对话
- ✅ **图片生成**：使用Cogview-3-Flash免费生成高质量图片
- ✅ **视频生成**：使用CogVideoX-Flash免费生成短视频
- ✅ **参数调节**：支持温度、最大token数等参数自定义
- ✅ **错误处理**：完善的错误提示和异常处理

## 支持的模型

### 🆓 免费对话模型
- `glm-4.5-flash` - GLM-4.5快速版（免费）⭐
- `glm-4.1v-thinking-flash` - GLM-4.1V思考版（免费）👁️**支持视觉**
- `glm-4-flash-250414` - GLM-4特定版本（免费）
- `glm-4v-flash` - GLM-4V视觉快速版（免费）👁️**支持视觉**⭐
- `glm-z1-flash` - GLM-Z1快速版（免费）

### 💰 付费对话模型
- `glm-4.5` - GLM-4.5标准版
- `glm-4` - GLM-4基础模型
- `glm-4-0520` - GLM-4特定版本
- `glm-4-plus` - GLM-4增强版
- `glm-4-air` - GLM-4轻量版
- `glm-4-airx` - GLM-4轻量增强版
- `glm-4-long` - GLM-4长文本版
- `glm-4-flashx` - GLM-4快速版
- `glm-4v` - GLM-4视觉模型 👁️**支持视觉**
- `glm-4v-plus` - GLM-4视觉增强版 👁️**支持视觉**

### 🎨 图片生成模型
- `cogview-3-flash` - Cogview-3快速版（免费）⭐

### 🎬 视频生成模型
- `cogvideox-flash` - CogVideoX快速版（免费）⭐

## 安装方法

### 1. 手动安装

1. 将插件文件夹复制到ComfyUI的 `custom_nodes` 目录下：
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url> zhipu-comfyui
   ```

2. 安装依赖包：
   ```bash
   cd zhipu-comfyui
   pip install -r requirements.txt
   ```

3. 重启ComfyUI

### 2. 通过ComfyUI Manager安装

在ComfyUI Manager中搜索"ZhipuAI"并安装。

## 使用方法

### 1. 获取API Key

1. 访问 [智谱AI开放平台](https://open.bigmodel.cn/)
2. 注册账号并登录
3. 在控制台中创建API Key
4. 复制API Key备用

### 2. 基本使用流程

#### 文本对话

1. 添加"智谱AI API配置"节点，输入您的API Key
2. 添加"智谱AI文本对话"节点
3. 连接API配置节点到文本对话节点
4. 在文本对话节点中输入您的问题
5. 执行工作流获得AI回复

#### 视觉对话

1. 添加"智谱AI API配置"节点，输入您的API Key
2. 添加"智谱AI视觉对话"节点
3. 连接API配置节点到视觉对话节点
4. 连接图片输入到视觉对话节点
5. 输入关于图片的问题
6. 执行工作流获得AI回复

#### 多轮对话

1. 使用"智谱AI对话历史"节点
2. 连接API配置和输入内容
3. 将历史输出连接回输入，实现连续对话

#### 图片生成

1. 添加"智谱AI API配置"节点，输入您的API Key
2. 添加"智谱AI图片生成"节点
3. 连接API配置节点到图片生成节点
4. 输入图片描述提示词
5. 选择图片尺寸和质量
6. 执行工作流获得图片URL

#### 视频生成

1. 添加"智谱AI API配置"节点，输入您的API Key
2. 添加"智谱AI视频生成"节点
3. 连接API配置节点到视频生成节点
4. 输入视频描述提示词
5. 可选：输入参考图片URL（图生视频）
6. 设置视频时长
7. 执行工作流获得视频URL

## 缓存绕过与随机种子（ComfyUI）

ComfyUI 会对相同输入进行缓存，导致“内容一样不重新运行”。本插件在如下节点新增 `seed` 参数，并将其透传为接口的 `request_id`，每次修改 `seed` 都会强制绕过缓存、触发重新执行：

- `ZhipuTextChat`（智谱AI文本对话）
- `ZhipuVisionChat`（智谱AI视觉对话）
- `ZhipuChatHistory`（智谱AI对话历史）

使用建议：
- 不需要随机性时保持 `seed=0`（不传 `request_id`），需要强制重跑时改一个整数即可。
- 与 `temperature/top_p` 等采样参数相互独立，不会影响生成质量，只用于“换一个请求 ID”。

可选调试：
- 启动前设置环境变量可打印请求载荷（自动截断图片base64）：
  ```bash
  export ZHIPU_DEBUG=1
  ```

## 节点说明

### ZhipuAPIConfig - 智谱AI API配置
**输入**：
- `api_key` (必需)：智谱AI的API Key

**输出**：
- `zhipu_client`：智谱AI客户端实例

### ZhipuTextChat - 智谱AI文本对话
**输入**：
- `zhipu_client` (必需)：API客户端实例
- `prompt` (必需)：用户输入的问题或对话内容
- `model` (必需)：选择使用的模型
- `temperature` (必需)：控制回复的随机性 (0.0-1.0)
- `max_tokens` (必需)：最大回复长度
- `system_prompt` (可选)：系统提示词

**输出**：
- `response`：AI生成的回复文本

### ZhipuVisionChat - 智谱AI视觉对话
**输入**：
- `zhipu_client` (必需)：API客户端实例
- `prompt` (必需)：关于图片的问题
- `image` (必需)：输入图片
- `model` (必需)：选择视觉模型
- `temperature` (必需)：控制回复的随机性
- `max_tokens` (必需)：最大回复长度
- `system_prompt` (可选)：系统提示词

**输出**：
- `response`：AI对图片的分析和回复

### ZhipuChatHistory - 智谱AI对话历史
**输入**：
- `zhipu_client` (必需)：API客户端实例
- `user_input` (必需)：当前用户输入
- `model` (必需)：选择使用的模型
- `temperature` (必需)：控制回复的随机性
- `max_tokens` (必需)：最大回复长度
- `system_prompt` (可选)：系统提示词
- `reset_history` (可选)：是否重置对话历史
- `chat_history` (可选)：历史对话记录

**输出**：
- `response`：AI的回复
- `chat_history`：更新后的对话历史

### ZhipuImageGeneration - 智谱AI图片生成
**输入**：
- `zhipu_client` (必需)：API客户端实例
- `prompt` (必需)：图片描述提示词
- `model` (必需)：选择图片生成模型（默认cogview-3-flash）
- `size` (必需)：图片尺寸（1024x1024, 768x1344, 1344x768等）
- `quality` (必需)：图片质量（standard, hd）
- `n` (必需)：生成图片数量（1-4张）

**输出**：
- `image_urls`：生成的图片URL列表（一行一个）
- `response_info`：生成信息和状态

### ZhipuVideoGeneration - 智谱AI视频生成
**输入**：
- `zhipu_client` (必需)：API客户端实例
- `prompt` (必需)：视频描述提示词
- `model` (必需)：选择视频生成模型（默认cogvideox-flash）
- `duration` (必需)：视频时长（2-10秒）
- `image_url` (可选)：参考图片URL（图生视频功能）

**输出**：
- `video_url`：生成的视频URL
- `response_info`：生成信息和状态

## 参数说明

- **temperature**: 控制输出的随机性，范围0.0-1.0
  - 0.0：输出最确定、最可预测
  - 1.0：输出最具创造性和随机性
  - 推荐值：0.7

- **max_tokens**: 限制AI回复的最大长度
  - 范围：1-8192
  - 推荐值：1024

- **top_p**: 核采样参数，控制候选词的概率分布
  - 范围：0.0-1.0
  - 默认值：0.9

## 注意事项

1. **API Key安全**：请妥善保管您的API Key，不要在公共场所或代码中暴露
2. **费用控制**：智谱AI按照token使用量计费，请注意控制使用量
3. **模型选择**：
   - 文本对话：推荐 `glm-4.5-flash`（免费）
   - 视觉理解：可选择 `glm-4v-flash` 或 `glm-4.1v-thinking-flash`（都免费且支持视觉）
   - 图片生成：使用 `cogview-3-flash`（免费）
   - 视频生成：使用 `cogvideox-flash`（免费）
4. **图片格式**：支持常见图片格式（PNG、JPEG、WEBP等）
5. **网络连接**：需要稳定的网络连接访问智谱AI API

## 错误排查

### 常见错误及解决方案

1. **API Key不能为空**
   - 检查是否正确输入了API Key
   - 确认API Key格式正确

2. **不支持的模型**
   - 确认选择的模型在支持列表中
   - 视觉任务必须使用支持视觉的模型：`glm-4v-flash`、`glm-4.1v-thinking-flash`、`glm-4v`、`glm-4v-plus`

3. **API调用失败**
   - 检查网络连接
   - 确认API Key有效且有足够余额
   - 检查请求参数是否正确

4. **图片处理失败**
   - 确认图片格式正确
   - 检查图片大小是否过大

## 更新日志

### v1.1.0
- ✨ 新增智谱AI图片生成节点（Cogview-3-Flash）
- ✨ 新增智谱AI视频生成节点（CogVideoX-Flash）
- 🆓 支持所有免费模型（Flash系列）
- 📝 更新模型列表，区分免费和付费模型
- 🔧 优化API客户端，支持图片和视频生成
- 📚 完善测试脚本，支持所有功能测试

### v1.0.0
- 初始版本发布
- 支持智谱AI文本对话
- 支持智谱AI视觉对话
- 支持多轮对话历史
- 完整的参数配置

## 技术支持

如遇到问题或需要帮助，请：

1. 查看本文档的错误排查部分
2. 检查ComfyUI控制台的错误日志
3. 确认智谱AI API服务状态
4. 联系开发者或提交Issue

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个插件！ 