# 尝试导入可用的模型类，忽略失败的导入
try:
    from .model import LlavaLlamaForCausalLM
except ImportError:
    pass

try:
    from .model import StreamOmniLlamaForCausalLM, StreamOmniConfig
except ImportError:
    pass
