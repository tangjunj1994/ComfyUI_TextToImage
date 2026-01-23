"""
ComfyUI Text to Image Plugin
使用BytePlus ModelArk Seedream 4.5/4.0模型进行文生图
参考文档: https://docs.byteplus.com/en/docs/ModelArk/1824121
"""

from .nodes.seedream_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Plugin metadata
WEB_DIRECTORY = "./web"
__version__ = "1.0.0"
