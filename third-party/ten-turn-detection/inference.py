#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TEN Turn Detector Inference Script

This script provides a simple command-line interface for the TEN Turn Detector,
allowing users to analyze text inputs and determine their conversation state.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TEN Turn Detector - Analyze conversation states in text inputs"
    )
    parser.add_argument(
        "--system", 
        type=str, 
        default="", 
        help="Optional system prompt to provide context"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="User input text to analyze"
    )
    return parser.parse_args()

def load_model(model_id):
    """
    Load the TEN Turn Detector model and tokenizer on GPU with optimizations.

    Args:
        model_id: HuggingFace model ID

    Returns:
        tuple: (model, tokenizer)
    """
    # 确定最佳设备
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16
        print(f"🚀 使用CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16  # MPS更适合float16
        print("🚀 使用Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        print("⚠️ 使用CPU，性能可能较慢")

    # 加载模型和分词器，优化GPU使用
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,  # 自动设备映射
        low_cpu_mem_usage=True,  # 降低CPU内存使用
        attn_implementation="flash_attention_2" if device == "cuda" else None  # 使用Flash Attention
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        use_fast=True  # 使用快速分词器
    )

    # 手动移动到设备（如果device_map没有处理）
    if device != "cuda":  # device_map="auto"已经处理了CUDA情况
        model = model.to(device)

    model.eval()

    # 启用编译优化（PyTorch 2.0+）
    if hasattr(torch, 'compile') and device == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("✅ 启用PyTorch编译优化")
        except Exception as e:
            print(f"⚠️ 编译优化失败: {e}")

    return model, tokenizer

def infer(model, tokenizer, system_prompt, user_input):
    """
    Run inference with the TEN Turn Detector with GPU optimizations.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        system_prompt: Optional system prompt for context
        user_input: User text to analyze

    Returns:
        str: The detected conversation state
    """
    inf_messages = [{"role":"system", "content":system_prompt}] + [{"role":"user", "content":user_input}]

    # 使用优化的分词
    input_ids = tokenizer.apply_chat_template(
        inf_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512  # 限制最大长度
    )

    # Move input to same device as model
    input_ids = input_ids.to(model.device)

    # 使用优化的推理设置
    with torch.no_grad():
        # 启用自动混合精度（如果支持）
        if model.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=1,
                    do_sample=False,  # 使用贪婪解码提高速度
                    num_beams=1,      # 单束搜索
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,   # 启用KV缓存
                    early_stopping=True
                )
        else:
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,  # 使用贪婪解码提高速度
                num_beams=1,      # 单束搜索
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,   # 启用KV缓存
                early_stopping=True
            )

    response = outputs[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True)
    return output

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Model ID on HuggingFace
    model_id = 'TEN-framework/TEN_Turn_Detector'
    
    # Load model and tokenizer
    print(f"Loading model from {model_id}...")
    model, tokenizer = load_model(model_id)
    
    # Run inference
    print(f"Running inference on: '{args.input}'")
    output = infer(model, tokenizer, args.system, args.input)
    
    # Print results
    print("\nResults:")
    print(f"Input: '{args.input}'")
    print(f"Turn Detection Result: '{output}'")
