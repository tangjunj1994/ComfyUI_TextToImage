# ComfyUI TextToImage - BytePlus Seedream 4.5

ComfyUI 自定义节点插件，使用 BytePlus ModelArk Seedream 4.5/4.0 模型进行文生图。

**使用官方 BytePlus Ark SDK 调用 API**

## 功能特点

- 🎨 **文生图 (Text-to-Image)**: 输入文本描述，生成高质量图片
- 🖼️ **图生图 (Image-to-Image)**: 使用参考图片 + 文本指令进行图片编辑
- 📦 **批量生成 (Batch Generate)**: 一次生成多张相关联的系列图片
- 🔀 **多图融合 (Multi-Image Blend)**: 多张图片融合生成新图片

## 支持的模型

| 模型名称 | Model ID | 说明 |
|---------|----------|------|
| Seedream 4.5 | `seedream-4-5-251128` | 最新最强，推荐使用 |
| Seedream 4.0 | `seedream-4-0-250828` | 平衡成本与质量 |

## 安装

### 方法1: 手动安装

1. 将 `ComfyUI_TextToImage` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录
2. 安装依赖：
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI_TextToImage
   pip install -r requirements.txt
   ```
3. 重启 ComfyUI

### 方法2: Git 克隆

```bash
cd ComfyUI/custom_nodes
git clone <repository_url> ComfyUI_TextToImage
cd ComfyUI_TextToImage
pip install -r requirements.txt
```

## 配置

### 获取 API Key

1. 访问 [BytePlus 控制台](https://console.byteplus.com/ark/apiKey) 获取 API Key
2. 在 [模型管理](https://console.byteplus.com/ark/openManagement) 启用 Seedream 模型服务

### 设置 API Key

有两种方式设置 API Key：

**方式1: 节点参数**
在节点的 `api_key` 输入框中直接输入

**方式2: 环境变量**
```bash
export ARK_API_KEY="your-api-key-here"
```

## 使用方法

### 文生图节点 (Seedream Text to Image)

基础使用：
1. 添加 `Seedream Text to Image (4.5)` 节点
2. 输入提示词 (prompt)
3. 设置 API Key
4. 选择模型和尺寸
5. 连接输出到 `Preview Image` 或 `Save Image` 节点

### 图生图节点 (Seedream Image to Image)

1. 添加 `Seedream Image to Image (4.5)` 节点
2. 连接输入图片
3. 输入编辑指令
4. 设置其他参数
5. 执行生成

### 批量生成节点 (Seedream Batch Generate)

1. 添加 `Seedream Batch Generate (4.5)` 节点
2. 输入系列图片的描述
3. 设置 `max_images` 参数
4. 输出为批量图片

### 多图融合节点 (Seedream Multi-Image Blend)

1. 添加 `Seedream Multi-Image Blend (4.5)` 节点
2. 连接2-3张参考图片
3. 输入融合指令
4. 执行生成

## 参数说明

### 通用参数

| 参数 | 类型 | 说明 |
|------|------|------|
| prompt | string | 图片描述文本 (英文效果更佳) |
| api_key | string | BytePlus API 密钥 |
| model | choice | 模型版本选择 |
| size | choice | 输出图片尺寸 |
| watermark | boolean | 是否添加水印 |
| api_base_url | string | API 端点 URL |

### 尺寸选项

- `512x512`, `768x768`, `1024x1024`
- `1280x720` (16:9), `720x1280` (9:16)
- `1920x1080` (全高清), `1080x1920` (竖版全高清)
- `2K` (2K分辨率)

## 提示词建议

1. 使用自然语言描述 **主体 + 动作 + 环境**
2. 如需美学效果，添加 **风格、颜色、光照、构图** 描述
3. 提示词建议不超过 600 个英文单词
4. 英文提示词效果优于中文

### 示例提示词

```
A beautiful sunset over the ocean with vibrant orange and purple colors, 
photorealistic style, shot on medium format camera, dramatic lighting
```

```
Vibrant close-up editorial portrait, model with piercing gaze, 
wearing a sculptural hat, rich color blocking, sharp focus on eyes, 
Vogue magazine cover aesthetic
```

## API 参考

本插件使用 BytePlus ModelArk Image Generation API，兼容 OpenAI 格式。

- API 端点: `https://ark.ap-southeast.bytepluses.com/api/v3/images/generations`
- 文档: [Seedream 4.0-4.5 Tutorial](https://docs.byteplus.com/en/docs/ModelArk/1824121)

## 计费说明

- Seedream 4.5/4.0: 查看 [Image Generation 定价](https://docs.byteplus.com/docs/ModelArk/1544106#c02be6ee)
- 新用户可获得 200 张免费额度

## 故障排除

### 常见错误

1. **API Key 无效**
   - 检查 API Key 是否正确
   - 确认模型服务已启用

2. **请求超时**
   - 检查网络连接
   - 批量生成可能需要更长时间

3. **图片尺寸不支持**
   - 使用预设尺寸选项
   - 或使用 "2K" 自动选择最佳尺寸

## 许可证

MIT License

## 相关链接

- [BytePlus ModelArk 文档](https://docs.byteplus.com/en/docs/ModelArk)
- [ComfyUI 官方仓库](https://github.com/comfyanonymous/ComfyUI)
- [Seedream 4.0-4.5 教程](https://docs.byteplus.com/en/docs/ModelArk/1824121)
