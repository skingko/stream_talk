import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    "stream_omni_llama": "StreamOmniLlamaForCausalLM, StreamOmniConfig",
}

# 成功导入的模型类将被添加到全局命名空间
_imported_models = {}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        # 使用exec导入模型类
        local_vars = {}
        exec(f"from .language_model.{model_name} import {model_classes}", globals(), local_vars)

        # 将成功导入的类添加到当前模块的全局命名空间
        for class_name in model_classes.split(", "):
            if class_name in local_vars:
                globals()[class_name] = local_vars[class_name]
                _imported_models[class_name] = model_name

        print(f"✅ Successfully imported {model_name}: {model_classes}")

    except Exception as e:
        print(f"Failed to import {model_name} from stream_omni.model.language_model.{model_name}. Error: {e}")
        continue
