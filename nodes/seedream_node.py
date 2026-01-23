"""
BytePlus Seedream Node for ComfyUI
使用BytePlus ModelArk Seedream 4.5/4.0模型进行文生图
参考文档: https://docs.byteplus.com/en/docs/ModelArk/1824121

使用BytePlus官方Ark SDK调用API
安装SDK: pip install byteplus-python-sdk-v2
"""

import os
import time
import base64
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, List, Union

import folder_paths
import torch

# 使用BytePlus官方Ark SDK
from byteplussdkarkruntime import Ark
from byteplussdkarkruntime.types.images.images import SequentialImageGenerationOptions

# API配置
BYTEPLUS_API_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"


class SeedreamTextToImage:
    """
    BytePlus Seedream 4.5/4.0 文生图节点
    使用BytePlus官方Ark SDK调用Seedream模型生成图片
    """
    
    # 支持的模型
    MODELS = [
        "seedream-4-5-251128",  # Seedream 4.5 (推荐)
        "seedream-4-0-250828",  # Seedream 4.0
    ]
    
    # 支持的图片尺寸
    SIZES = [
        "512x512",
        "768x768", 
        "1024x1024",
        "1280x720",
        "720x1280",
        "1920x1080",
        "1080x1920",
        "2560x1440",
        "1440x2560",
        "2K",  # 2K分辨率
    ]
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful sunset over the ocean with vibrant orange and purple colors, photorealistic style",
                    "tooltip": "图片描述文本 (英文效果更佳)"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BytePlus API密钥 (也可通过环境变量ARK_API_KEY设置)"
                }),
                "model": (cls.MODELS, {
                    "default": "seedream-4-5-251128",
                    "tooltip": "选择Seedream模型版本"
                }),
                "size": (cls.SIZES, {
                    "default": "1024x1024",
                    "tooltip": "生成图片的尺寸"
                }),
            },
            "optional": {
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": BYTEPLUS_API_BASE_URL,
                    "tooltip": "API基础URL"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                    "tooltip": "随机种子 (-1为随机)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "status_message")
    FUNCTION = "generate_image"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    
    def generate_image(self, prompt: str, api_key: str, model: str, size: str,
                       api_base_url: str = BYTEPLUS_API_BASE_URL,
                       watermark: bool = False, seed: int = -1):
        """
        调用Seedream API生成图片 (使用BytePlus Ark SDK)
        """
        try:
            # 获取API密钥
            effective_api_key = api_key if api_key else os.environ.get("ARK_API_KEY", "")
            if not effective_api_key:
                error_msg = "❌ 错误：未提供API密钥。请在节点中输入或设置环境变量ARK_API_KEY"
                empty_image = self._create_empty_image()
                return (empty_image, "", error_msg)
            
            # 创建Ark客户端
            client = Ark(
                base_url=api_base_url,
                api_key=effective_api_key,
            )
            
            print(f"🎨 正在调用Seedream API生成图片...")
            print(f"   模型: {model}")
            print(f"   尺寸: {size}")
            print(f"   提示词: {prompt[:100]}...")
            
            start_time = time.time()
            
            # 构建参数
            generate_params = {
                "model": model,
                "prompt": prompt,
                "size": size,
                "response_format": "url",
                "watermark": watermark,
            }
            
            # 添加种子
            if seed >= 0:
                generate_params["seed"] = seed
            
            # 调用API生成图片 (使用Ark SDK)
            images_response = client.images.generate(**generate_params)
            
            elapsed_time = time.time() - start_time
            
            # 获取图片URL
            if images_response.data and len(images_response.data) > 0:
                image_url = images_response.data[0].url
                
                if image_url:
                    # 下载图片
                    image_tensor = self._download_and_convert_image(image_url)
                    
                    status_message = (
                        f"✅ 图片生成成功！\n"
                        f"   模型: {model}\n"
                        f"   尺寸: {size}\n"
                        f"   耗时: {elapsed_time:.2f}秒"
                    )
                    print(status_message)
                    
                    return (image_tensor, image_url, status_message)
            
            error_msg = f"❌ 无法获取图片URL"
            print(error_msg)
            empty_image = self._create_empty_image()
            return (empty_image, "", error_msg)
            
        except Exception as e:
            error_msg = f"❌ 生成图片时发生错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            empty_image = self._create_empty_image()
            return (empty_image, "", error_msg)
    
    def _download_and_convert_image(self, url: str) -> torch.Tensor:
        """下载图片并转换为ComfyUI张量格式"""
        print(f"📥 正在下载图片: {url[:80]}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        # 转换为numpy数组，然后转换为torch张量
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return image_tensor
    
    def _create_empty_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        """创建空白图片（用于错误情况）"""
        empty = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(empty)[None,]


class SeedreamImageToImage:
    """
    BytePlus Seedream 图生图节点
    使用输入图片和文本提示生成新图片
    """
    
    MODELS = SeedreamTextToImage.MODELS
    SIZES = SeedreamTextToImage.SIZES
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入参考图片"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Keep the subject and composition, change the style to oil painting",
                    "tooltip": "编辑指令 (描述如何修改图片)"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BytePlus API密钥"
                }),
                "model": (cls.MODELS, {
                    "default": "seedream-4-5-251128",
                    "tooltip": "选择Seedream模型版本"
                }),
                "size": (cls.SIZES, {
                    "default": "1024x1024",
                    "tooltip": "生成图片的尺寸"
                }),
            },
            "optional": {
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": BYTEPLUS_API_BASE_URL,
                    "tooltip": "API基础URL"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "status_message")
    FUNCTION = "generate_image"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    
    def generate_image(self, image: torch.Tensor, prompt: str, api_key: str, 
                       model: str, size: str,
                       api_base_url: str = BYTEPLUS_API_BASE_URL,
                       watermark: bool = False):
        """
        调用Seedream API进行图生图 (使用BytePlus Ark SDK)
        """
        try:
            # 获取API密钥
            effective_api_key = api_key if api_key else os.environ.get("ARK_API_KEY", "")
            if not effective_api_key:
                error_msg = "❌ 错误：未提供API密钥"
                empty_image = self._create_empty_image()
                return (empty_image, "", error_msg)
            
            # 将输入图片转换为base64 data URL
            image_data_url = self._tensor_to_data_url(image)
            
            # 创建Ark客户端
            client = Ark(
                base_url=api_base_url,
                api_key=effective_api_key,
            )
            
            print(f"🎨 正在调用Seedream API进行图生图...")
            print(f"   模型: {model}")
            print(f"   尺寸: {size}")
            print(f"   提示词: {prompt[:100]}...")
            
            start_time = time.time()
            
            # 调用API (使用Ark SDK)
            images_response = client.images.generate(
                model=model,
                prompt=prompt,
                image=image_data_url,
                size=size,
                response_format="url",
                watermark=watermark,
            )
            
            elapsed_time = time.time() - start_time
            
            # 获取图片URL
            if images_response.data and len(images_response.data) > 0:
                image_url = images_response.data[0].url
                
                if image_url:
                    image_tensor = self._download_and_convert_image(image_url)
                    
                    status_message = (
                        f"✅ 图生图成功！\n"
                        f"   模型: {model}\n"
                        f"   耗时: {elapsed_time:.2f}秒"
                    )
                    print(status_message)
                    
                    return (image_tensor, image_url, status_message)
            
            error_msg = f"❌ 无法获取图片URL"
            print(error_msg)
            empty_image = self._create_empty_image()
            return (empty_image, "", error_msg)
            
        except Exception as e:
            error_msg = f"❌ 图生图时发生错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            empty_image = self._create_empty_image()
            return (empty_image, "", error_msg)
    
    def _tensor_to_data_url(self, image_tensor: torch.Tensor) -> str:
        """将ComfyUI图片张量转换为data URL"""
        # 取第一张图片
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        # 转换为numpy数组
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # 转换为PIL Image
        image = Image.fromarray(image_np)
        
        # 保存为base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{b64_data}"
    
    def _download_and_convert_image(self, url: str) -> torch.Tensor:
        """下载图片并转换为ComfyUI张量格式"""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return image_tensor
    
    def _create_empty_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        """创建空白图片"""
        empty = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(empty)[None,]


class SeedreamTextToBatchImage:
    """
    BytePlus Seedream 文字批量生成图片节点
    Text-to-Batch-Image（Text Input, Batch Image Output)
    参考: https://docs.byteplus.com/en/docs/ModelArk/1824121#batch-image-output
    """
    
    MODELS = SeedreamTextToImage.MODELS
    SIZES = SeedreamTextToImage.SIZES
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate a series of 4 coherent illustrations focusing on the same corner of a courtyard across the four seasons, presented in a unified style that captures the unique colors, elements, and atmosphere of each season.",
                    "tooltip": "批量生成提示词 (描述要生成的系列图片)"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BytePlus API密钥"
                }),
                "model": (cls.MODELS, {
                    "default": "seedream-4-5-251128",
                    "tooltip": "选择Seedream模型版本"
                }),
                "size": (cls.SIZES, {
                    "default": "2K",
                    "tooltip": "生成图片的尺寸"
                }),
                "max_images": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "最大生成图片数量"
                }),
            },
            "optional": {
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": BYTEPLUS_API_BASE_URL,
                    "tooltip": "API基础URL"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status_message")
    FUNCTION = "generate_batch"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    
    def generate_batch(self, prompt: str, api_key: str, model: str, size: str,
                       max_images: int = 4,
                       api_base_url: str = BYTEPLUS_API_BASE_URL,
                       watermark: bool = False):
        """
        调用Seedream API批量生成图片 (使用BytePlus Ark SDK)
        Text-to-Batch-Image
        """
        try:
            # 获取API密钥
            effective_api_key = api_key if api_key else os.environ.get("ARK_API_KEY", "")
            if not effective_api_key:
                error_msg = "❌ 错误：未提供API密钥"
                empty_image = self._create_empty_image()
                return (empty_image, error_msg)
            
            # 创建Ark客户端
            client = Ark(
                base_url=api_base_url,
                api_key=effective_api_key,
            )

            prompt = f"Generate {max_images} images based on the prompt: {prompt}"
            
            print(f"🎨 正在调用Seedream API批量生成图片 (Text-to-Batch-Image)...")
            print(f"   模型: {model}")
            print(f"   尺寸: {size}")
            print(f"   最大图片数: {max_images}")
            print(f"   提示词: {prompt}...")
            
            start_time = time.time()
            
            # 调用API - 按照官方文档格式
            images_response = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                sequential_image_generation="auto",
                sequential_image_generation_options=SequentialImageGenerationOptions(max_images=max_images),
                response_format="url",
                watermark=watermark,
            )
            
            elapsed_time = time.time() - start_time
            
            # 下载所有图片
            if images_response.data and len(images_response.data) > 0:
                print(images_response.data)
                images = []
                for idx, image_data in enumerate(images_response.data):
                    image_url = image_data.url
                    image_size = getattr(image_data, 'size', 'unknown')
                    if image_url:
                        print(f"📥 下载图片 {idx + 1}/{len(images_response.data)} (URL: {image_url[:60]}..., Size: {image_size})")
                        image_tensor = self._download_and_convert_image(image_url)
                        images.append(image_tensor)
                
                if images:
                    # 合并所有图片为batch
                    batch_tensor = torch.cat(images, dim=0)
                    
                    status_message = (
                        f"✅ 批量生成成功！\n"
                        f"   生成图片数: {len(images)}\n"
                        f"   模型: {model}\n"
                        f"   耗时: {elapsed_time:.2f}秒"
                    )
                    print(status_message)
                    
                    return (batch_tensor, status_message)
            
            error_msg = f"❌ 无法获取图片"
            print(error_msg)
            empty_image = self._create_empty_image()
            return (empty_image, error_msg)
            
        except Exception as e:
            error_msg = f"❌ 批量生成时发生错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            empty_image = self._create_empty_image()
            return (empty_image, error_msg)
    
    def _download_and_convert_image(self, url: str) -> torch.Tensor:
        """下载图片并转换为ComfyUI张量格式"""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return image_tensor
    
    def _create_empty_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        """创建空白图片"""
        empty = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(empty)[None,]


class SeedreamImageToBatchImage:
    """
    BytePlus Seedream 图生批量图片节点
    Image-to-Batch-Image (Single Image Input, Batch Image Output)
    参考: https://docs.byteplus.com/en/docs/ModelArk/1824121#batch-image-output
    """
    
    MODELS = SeedreamTextToImage.MODELS
    SIZES = SeedreamTextToImage.SIZES
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入参考图片"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Using this LOGO as a reference, create a visual design system for an outdoor sports brand named GREEN, including packaging bags, hats, cards, lanyards, etc. Main visual tone is green, with a fun, simple, and modern style.",
                    "tooltip": "批量生成提示词"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BytePlus API密钥"
                }),
                "model": (cls.MODELS, {
                    "default": "seedream-4-5-251128",
                    "tooltip": "选择Seedream模型版本"
                }),
                "size": (cls.SIZES, {
                    "default": "2K",
                    "tooltip": "生成图片的尺寸"
                }),
                "max_images": ("INT", {
                    "default": 4,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "最大生成图片数量"
                }),
            },
            "optional": {
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": BYTEPLUS_API_BASE_URL,
                    "tooltip": "API基础URL"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status_message")
    FUNCTION = "generate_batch"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    
    def generate_batch(self, image: torch.Tensor, prompt: str, api_key: str, 
                       model: str, size: str, max_images: int = 4,
                       api_base_url: str = BYTEPLUS_API_BASE_URL,
                       watermark: bool = False):
        """
        调用Seedream API进行图生批量图片 (使用BytePlus Ark SDK)
        Image-to-Batch-Image
        """
        try:
            # 获取API密钥
            effective_api_key = api_key if api_key else os.environ.get("ARK_API_KEY", "")
            if not effective_api_key:
                error_msg = "❌ 错误：未提供API密钥"
                empty_image = self._create_empty_image()
                return (empty_image, error_msg)
            
            # 将输入图片转换为data URL
            image_data_url = self._tensor_to_data_url(image)
            
            # 创建Ark客户端
            client = Ark(
                base_url=api_base_url,
                api_key=effective_api_key,
            )
            
            print(f"🎨 正在调用Seedream API批量生成图片 (Image-to-Batch-Image)...")
            print(f"   模型: {model}")
            print(f"   尺寸: {size}")
            print(f"   最大图片数: {max_images}")
            print(f"   提示词: {prompt[:100]}...")
            
            start_time = time.time()
            
            # 调用API - 按照官方文档格式
            images_response = client.images.generate(
                model=model,
                prompt=prompt,
                image=image_data_url,
                size=size,
                sequential_image_generation="auto",
                sequential_image_generation_options=SequentialImageGenerationOptions(max_images=max_images),
                response_format="url",
                watermark=watermark,
            )
            
            elapsed_time = time.time() - start_time
            
            # 下载所有图片
            if images_response.data and len(images_response.data) > 0:
                images = []
                for idx, image_data in enumerate(images_response.data):
                    image_url = image_data.url
                    image_size = getattr(image_data, 'size', 'unknown')
                    if image_url:
                        print(f"📥 下载图片 {idx + 1}/{len(images_response.data)} (Size: {image_size})")
                        image_tensor = self._download_and_convert_image(image_url)
                        images.append(image_tensor)
                
                if images:
                    # 合并所有图片为batch
                    batch_tensor = torch.cat(images, dim=0)
                    
                    status_message = (
                        f"✅ 图生批量图片成功！\n"
                        f"   生成图片数: {len(images)}\n"
                        f"   模型: {model}\n"
                        f"   耗时: {elapsed_time:.2f}秒"
                    )
                    print(status_message)
                    
                    return (batch_tensor, status_message)
            
            error_msg = f"❌ 无法获取图片"
            print(error_msg)
            empty_image = self._create_empty_image()
            return (empty_image, error_msg)
            
        except Exception as e:
            error_msg = f"❌ 图生批量图片时发生错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            empty_image = self._create_empty_image()
            return (empty_image, error_msg)
    
    def _tensor_to_data_url(self, image_tensor: torch.Tensor) -> str:
        """将ComfyUI图片张量转换为data URL"""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{b64_data}"
    
    def _download_and_convert_image(self, url: str) -> torch.Tensor:
        """下载图片并转换为ComfyUI张量格式"""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return image_tensor
    
    def _create_empty_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        """创建空白图片"""
        empty = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(empty)[None,]


class SeedreamMultiImageToBatchImage:
    """
    BytePlus Seedream 多图生批量图片节点
    Multi-Image-to-Batch-Image (Multi-Image Input, Batch-Image Output)
    参考: https://docs.byteplus.com/en/docs/ModelArk/1824121#batch-image-output
    """
    
    MODELS = SeedreamTextToImage.MODELS
    SIZES = SeedreamTextToImage.SIZES
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {
                    "tooltip": "第一张参考图片"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "第二张参考图片"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Generate 3 images of a girl and a cow plushie happily riding a roller coaster in an amusement park, depicting morning, noon, and night.",
                    "tooltip": "批量生成提示词"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BytePlus API密钥"
                }),
                "model": (cls.MODELS, {
                    "default": "seedream-4-5-251128",
                    "tooltip": "选择Seedream模型版本"
                }),
                "size": (cls.SIZES, {
                    "default": "2K",
                    "tooltip": "生成图片的尺寸"
                }),
                "max_images": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 8,
                    "step": 1,
                    "tooltip": "最大生成图片数量"
                }),
            },
            "optional": {
                "image3": ("IMAGE", {
                    "tooltip": "第三张参考图片 (可选)"
                }),
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": BYTEPLUS_API_BASE_URL,
                    "tooltip": "API基础URL"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "status_message")
    FUNCTION = "generate_batch"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    
    def generate_batch(self, image1: torch.Tensor, image2: torch.Tensor, 
                       prompt: str, api_key: str, model: str, size: str,
                       max_images: int = 3,
                       image3: Optional[torch.Tensor] = None,
                       api_base_url: str = BYTEPLUS_API_BASE_URL,
                       watermark: bool = False):
        """
        调用Seedream API进行多图生批量图片 (使用BytePlus Ark SDK)
        Multi-Image-to-Batch-Image
        """
        try:
            # 获取API密钥
            effective_api_key = api_key if api_key else os.environ.get("ARK_API_KEY", "")
            if not effective_api_key:
                error_msg = "❌ 错误：未提供API密钥"
                empty_image = self._create_empty_image()
                return (empty_image, error_msg)
            
            # 将图片转换为data URL列表
            images_list = [
                self._tensor_to_data_url(image1),
                self._tensor_to_data_url(image2),
            ]
            
            if image3 is not None:
                images_list.append(self._tensor_to_data_url(image3))
            
            # 创建Ark客户端
            client = Ark(
                base_url=api_base_url,
                api_key=effective_api_key,
            )
            
            print(f"🎨 正在调用Seedream API批量生成图片 (Multi-Image-to-Batch-Image)...")
            print(f"   模型: {model}")
            print(f"   输入图片数: {len(images_list)}")
            print(f"   尺寸: {size}")
            print(f"   最大图片数: {max_images}")
            print(f"   提示词: {prompt[:100]}...")
            
            start_time = time.time()
            
            # 调用API - 按照官方文档格式
            images_response = client.images.generate(
                model=model,
                prompt=prompt,
                image=images_list,
                size=size,
                sequential_image_generation="auto",
                sequential_image_generation_options=SequentialImageGenerationOptions(max_images=max_images),
                response_format="url",
                watermark=watermark,
            )
            
            elapsed_time = time.time() - start_time
            
            # 下载所有图片
            if images_response.data and len(images_response.data) > 0:
                images = []
                for idx, image_data in enumerate(images_response.data):
                    image_url = image_data.url
                    image_size = getattr(image_data, 'size', 'unknown')
                    if image_url:
                        print(f"📥 下载图片 {idx + 1}/{len(images_response.data)} (Size: {image_size})")
                        image_tensor = self._download_and_convert_image(image_url)
                        images.append(image_tensor)
                
                if images:
                    # 合并所有图片为batch
                    batch_tensor = torch.cat(images, dim=0)
                    
                    status_message = (
                        f"✅ 多图生批量图片成功！\n"
                        f"   生成图片数: {len(images)}\n"
                        f"   模型: {model}\n"
                        f"   耗时: {elapsed_time:.2f}秒"
                    )
                    print(status_message)
                    
                    return (batch_tensor, status_message)
            
            error_msg = f"❌ 无法获取图片"
            print(error_msg)
            empty_image = self._create_empty_image()
            return (empty_image, error_msg)
            
        except Exception as e:
            error_msg = f"❌ 多图生批量图片时发生错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            empty_image = self._create_empty_image()
            return (empty_image, error_msg)
    
    def _tensor_to_data_url(self, image_tensor: torch.Tensor) -> str:
        """将ComfyUI图片张量转换为data URL"""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{b64_data}"
    
    def _download_and_convert_image(self, url: str) -> torch.Tensor:
        """下载图片并转换为ComfyUI张量格式"""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return image_tensor
    
    def _create_empty_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        """创建空白图片"""
        empty = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(empty)[None,]


class SeedreamMultiImageBlend:
    """
    BytePlus Seedream 多图融合节点
    使用多张参考图片融合生成单张新图片 (非批量)
    """
    
    MODELS = SeedreamTextToImage.MODELS
    SIZES = SeedreamTextToImage.SIZES
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {
                    "tooltip": "第一张参考图片"
                }),
                "image2": ("IMAGE", {
                    "tooltip": "第二张参考图片"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Replace the clothing in image 1 with the outfit from image 2",
                    "tooltip": "融合指令"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "BytePlus API密钥"
                }),
                "model": (cls.MODELS, {
                    "default": "seedream-4-5-251128",
                    "tooltip": "选择Seedream模型版本"
                }),
                "size": (cls.SIZES, {
                    "default": "2K",
                    "tooltip": "生成图片的尺寸"
                }),
            },
            "optional": {
                "image3": ("IMAGE", {
                    "tooltip": "第三张参考图片 (可选)"
                }),
                "api_base_url": ("STRING", {
                    "multiline": False,
                    "default": BYTEPLUS_API_BASE_URL,
                    "tooltip": "API基础URL"
                }),
                "watermark": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否添加水印"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "image_url", "status_message")
    FUNCTION = "blend_images"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True
    
    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, 
                     prompt: str, api_key: str, model: str, size: str,
                     image3: Optional[torch.Tensor] = None,
                     api_base_url: str = BYTEPLUS_API_BASE_URL,
                     watermark: bool = False):
        """
        调用Seedream API进行多图融合 (使用BytePlus Ark SDK)
        """
        try:
            # 获取API密钥
            effective_api_key = api_key if api_key else os.environ.get("ARK_API_KEY", "")
            if not effective_api_key:
                error_msg = "❌ 错误：未提供API密钥"
                empty_image = self._create_empty_image()
                return (empty_image, "", error_msg)
            
            # 将图片转换为data URL列表
            images_list = [
                self._tensor_to_data_url(image1),
                self._tensor_to_data_url(image2),
            ]
            
            if image3 is not None:
                images_list.append(self._tensor_to_data_url(image3))
            
            # 创建Ark客户端
            client = Ark(
                base_url=api_base_url,
                api_key=effective_api_key,
            )
            
            print(f"🎨 正在调用Seedream API进行多图融合...")
            print(f"   模型: {model}")
            print(f"   输入图片数: {len(images_list)}")
            print(f"   提示词: {prompt[:100]}...")
            
            start_time = time.time()
            
            # 调用API (使用Ark SDK) - sequential_image_generation="disabled" 表示单图输出
            images_response = client.images.generate(
                model=model,
                prompt=prompt,
                image=images_list,
                size=size,
                sequential_image_generation="disabled",
                response_format="url",
                watermark=watermark,
            )
            
            elapsed_time = time.time() - start_time
            
            # 获取图片URL
            if images_response.data and len(images_response.data) > 0:
                image_url = images_response.data[0].url
                
                if image_url:
                    image_tensor = self._download_and_convert_image(image_url)
                    
                    status_message = (
                        f"✅ 多图融合成功！\n"
                        f"   模型: {model}\n"
                        f"   耗时: {elapsed_time:.2f}秒"
                    )
                    print(status_message)
                    
                    return (image_tensor, image_url, status_message)
            
            error_msg = f"❌ 无法获取图片URL"
            print(error_msg)
            empty_image = self._create_empty_image()
            return (empty_image, "", error_msg)
            
        except Exception as e:
            error_msg = f"❌ 多图融合时发生错误: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            empty_image = self._create_empty_image()
            return (empty_image, "", error_msg)
    
    def _tensor_to_data_url(self, image_tensor: torch.Tensor) -> str:
        """将ComfyUI图片张量转换为data URL"""
        if len(image_tensor.shape) == 4:
            image_tensor = image_tensor[0]
        
        image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        b64_data = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{b64_data}"
    
    def _download_and_convert_image(self, url: str) -> torch.Tensor:
        """下载图片并转换为ComfyUI张量格式"""
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        image = image.convert("RGB")
        
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        return image_tensor
    
    def _create_empty_image(self, width: int = 512, height: int = 512) -> torch.Tensor:
        """创建空白图片"""
        empty = np.zeros((height, width, 3), dtype=np.float32)
        return torch.from_numpy(empty)[None,]


# 节点映射
NODE_CLASS_MAPPINGS = {
    "SeedreamTextToImage": SeedreamTextToImage,
    "SeedreamImageToImage": SeedreamImageToImage,
    "SeedreamTextToBatchImage": SeedreamTextToBatchImage,
    "SeedreamImageToBatchImage": SeedreamImageToBatchImage,
    "SeedreamMultiImageToBatchImage": SeedreamMultiImageToBatchImage,
    "SeedreamMultiImageBlend": SeedreamMultiImageBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeedreamTextToImage": "Seedream Text to Image (4.5)",
    "SeedreamImageToImage": "Seedream Image to Image (4.5)",
    "SeedreamTextToBatchImage": "Seedream Text to Batch Image (4.5)",
    "SeedreamImageToBatchImage": "Seedream Image to Batch Image (4.5)",
    "SeedreamMultiImageToBatchImage": "Seedream Multi-Image to Batch Image (4.5)",
    "SeedreamMultiImageBlend": "Seedream Multi-Image Blend (4.5)",
}
