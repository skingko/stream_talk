#!/usr/bin/env python3
"""
Fish Speech Simulator
用于测试和备用的Fish Speech模拟器
基于基准测试结果提供真实的性能模拟
"""

import asyncio
import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator

logger = logging.getLogger(__name__)

class FishSpeechSimulator:
    """Fish Speech模拟器"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = device
        self.is_initialized = False
        
        # 基于基准测试的性能配置
        self.performance_config = {
            "mps": {
                "rtf_factor": 0.54,  # 基准测试平均RTF
                "min_rtf": 0.28,     # 最佳RTF
                "max_rtf": 0.77,     # 最差RTF
                "description": "Apple Silicon MPS"
            },
            "cuda": {
                "rtf_factor": 0.3,   # CUDA预期性能
                "min_rtf": 0.15,
                "max_rtf": 0.5,
                "description": "NVIDIA GPU"
            },
            "cpu": {
                "rtf_factor": 2.5,   # CPU性能
                "min_rtf": 2.0,
                "max_rtf": 3.0,
                "description": "CPU处理"
            }
        }
        
        # 获取当前设备的性能配置
        self.perf = self.performance_config.get(device, self.performance_config["cpu"])
        
        # 情感支持
        self.supported_emotions = [
            "neutral", "happy", "sad", "angry", "excited", "surprised",
            "joyful", "confident", "whispering", "shouting", "laughing",
            "chuckling", "sobbing", "sighing"
        ]
    
    async def initialize(self):
        """初始化模拟器"""
        try:
            logger.info("🎭 初始化Fish Speech模拟器...")
            
            # 检查模型文件
            if not self._check_model_files():
                logger.warning("⚠️ 模型文件不完整，使用模拟模式")
            
            # 模拟加载时间
            model_size = self._get_model_size()
            load_time = model_size * 0.5  # 每GB约0.5秒
            await asyncio.sleep(min(load_time, 3.0))
            
            self.is_initialized = True
            
            logger.info(f"✅ Fish Speech模拟器初始化完成")
            logger.info(f"🔧 设备: {self.device} ({self.perf['description']})")
            logger.info(f"🎯 预期RTF: {self.perf['rtf_factor']:.3f} (范围: {self.perf['min_rtf']:.3f}-{self.perf['max_rtf']:.3f})")
            
        except Exception as e:
            logger.error(f"❌ 模拟器初始化失败: {e}")
            raise
    
    def _check_model_files(self) -> bool:
        """检查模型文件"""
        required_files = ["model.pth", "codec.pth", "config.json", "tokenizer.tiktoken"]
        
        for file in required_files:
            file_path = self.model_path / file
            if not file_path.exists():
                return False
        
        return True
    
    def _get_model_size(self) -> float:
        """获取模型大小（GB）"""
        try:
            total_size = 0
            for file in ["model.pth", "codec.pth"]:
                file_path = self.model_path / file
                if file_path.exists():
                    total_size += file_path.stat().st_size
            
            return total_size / (1024**3)  # 转换为GB
        except:
            return 3.36  # 默认大小
    
    async def synthesize(self, text: str, emotion: str = "neutral", **kwargs) -> np.ndarray:
        """批量语音合成"""
        if not self.is_initialized:
            raise RuntimeError("模拟器未初始化")
        
        try:
            logger.info(f"🎵 模拟合成: {text[:50]}...")
            
            # 计算处理时间（基于真实基准测试）
            estimated_duration = len(text) * 0.15  # 中文约每字符0.15秒
            
            # 添加随机变化（±20%）
            variation = np.random.uniform(0.8, 1.2)
            rtf = self.perf['rtf_factor'] * variation
            
            # 确保在合理范围内
            rtf = max(self.perf['min_rtf'], min(rtf, self.perf['max_rtf']))
            
            processing_time = estimated_duration * rtf
            
            # 模拟处理延迟
            await asyncio.sleep(min(processing_time, 5.0))
            
            # 生成高质量模拟音频
            audio_data = self._generate_realistic_audio(text, emotion, estimated_duration)
            
            logger.info(f"✅ 模拟合成完成，RTF: {rtf:.3f}, 处理时间: {processing_time:.3f}s")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ 模拟合成失败: {e}")
            raise
    
    async def synthesize_streaming(self, text: str, emotion: str = "neutral", **kwargs) -> AsyncGenerator[np.ndarray, None]:
        """流式语音合成"""
        if not self.is_initialized:
            raise RuntimeError("模拟器未初始化")
        
        try:
            logger.info(f"🌊 模拟流式合成: {text[:50]}...")
            
            # 计算总时长和块数
            estimated_duration = len(text) * 0.15
            chunk_duration = 0.1  # 100ms块
            num_chunks = int(estimated_duration / chunk_duration)
            
            # 计算每块的处理时间
            rtf = self.perf['rtf_factor'] * np.random.uniform(0.8, 1.2)
            rtf = max(self.perf['min_rtf'], min(rtf, self.perf['max_rtf']))
            
            chunk_processing_time = chunk_duration * rtf
            
            for i in range(num_chunks):
                # 模拟处理延迟
                await asyncio.sleep(min(chunk_processing_time, 0.2))
                
                # 生成音频块
                chunk = self._generate_audio_chunk(emotion, chunk_duration, i, num_chunks)
                yield chunk
                
            logger.info(f"✅ 模拟流式合成完成，RTF: {rtf:.3f}")
            
        except Exception as e:
            logger.error(f"❌ 模拟流式合成失败: {e}")
            raise
    
    def _generate_realistic_audio(self, text: str, emotion: str, duration: float) -> np.ndarray:
        """生成真实的模拟音频"""
        sample_rate = 22050
        samples = int(sample_rate * duration)
        
        # 基础频率（根据情感调整）
        base_freq = self._get_emotion_frequency(emotion)
        
        # 生成时间轴
        t = np.linspace(0, duration, samples)
        
        # 生成复合音频信号
        audio = np.zeros(samples)
        
        # 基础音调
        audio += 0.3 * np.sin(2 * np.pi * base_freq * t)
        
        # 谐波
        audio += 0.15 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * base_freq * 3 * t)
        
        # 情感调制
        emotion_mod = self._get_emotion_modulation(emotion, t)
        audio *= emotion_mod
        
        # 添加自然变化
        frequency_variation = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)  # 0.5Hz变化
        audio *= frequency_variation
        
        # 添加轻微噪声（模拟自然语音）
        noise = 0.05 * np.random.randn(samples)
        audio += noise
        
        # 应用包络（避免突然开始/结束）
        envelope = self._apply_envelope(audio, sample_rate)
        audio *= envelope
        
        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def _generate_audio_chunk(self, emotion: str, duration: float, chunk_idx: int, total_chunks: int) -> np.ndarray:
        """生成音频块"""
        sample_rate = 22050
        samples = int(sample_rate * duration)
        
        # 基础频率
        base_freq = self._get_emotion_frequency(emotion)
        
        # 时间轴
        t = np.linspace(0, duration, samples)
        
        # 生成音频块
        audio = 0.3 * np.sin(2 * np.pi * base_freq * t)
        
        # 添加进度相关的变化
        progress = chunk_idx / total_chunks
        pitch_variation = 1 + 0.2 * np.sin(2 * np.pi * progress)
        audio *= pitch_variation
        
        # 情感调制
        emotion_mod = self._get_emotion_modulation(emotion, t)
        audio *= emotion_mod
        
        # 添加噪声
        noise = 0.03 * np.random.randn(samples)
        audio += noise
        
        # 归一化
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def _get_emotion_frequency(self, emotion: str) -> float:
        """根据情感获取基础频率"""
        emotion_freq_map = {
            "neutral": 220,
            "happy": 280,
            "excited": 320,
            "joyful": 300,
            "surprised": 350,
            "sad": 180,
            "angry": 160,
            "confident": 240,
            "whispering": 200,
            "shouting": 400,
            "laughing": 350,
            "chuckling": 260,
            "sobbing": 150,
            "sighing": 170
        }
        
        return emotion_freq_map.get(emotion, 220)
    
    def _get_emotion_modulation(self, emotion: str, t: np.ndarray) -> np.ndarray:
        """根据情感获取调制信号"""
        if emotion == "happy" or emotion == "joyful":
            # 快乐：轻快的调制
            return 1 + 0.2 * np.sin(2 * np.pi * 3 * t)
        elif emotion == "excited":
            # 兴奋：快速变化
            return 1 + 0.3 * np.sin(2 * np.pi * 5 * t)
        elif emotion == "sad" or emotion == "sobbing":
            # 悲伤：缓慢下降
            return 1 - 0.1 * t / np.max(t)
        elif emotion == "angry":
            # 愤怒：尖锐变化
            return 1 + 0.4 * np.sin(2 * np.pi * 7 * t)
        elif emotion == "whispering":
            # 耳语：低幅度
            return 0.5 * np.ones_like(t)
        elif emotion == "shouting":
            # 喊叫：高幅度
            return 1.5 * np.ones_like(t)
        else:
            # 中性：稳定
            return np.ones_like(t)
    
    def _apply_envelope(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """应用音频包络"""
        length = len(audio)
        envelope = np.ones(length)
        
        # 淡入（50ms）
        fade_in_samples = int(0.05 * sample_rate)
        if fade_in_samples < length:
            envelope[:fade_in_samples] = np.linspace(0, 1, fade_in_samples)
        
        # 淡出（50ms）
        fade_out_samples = int(0.05 * sample_rate)
        if fade_out_samples < length:
            envelope[-fade_out_samples:] = np.linspace(1, 0, fade_out_samples)
        
        return envelope
    
    def get_supported_emotions(self) -> list:
        """获取支持的情感列表"""
        return self.supported_emotions.copy()
    
    def get_performance_info(self) -> dict:
        """获取性能信息"""
        return {
            "device": self.device,
            "description": self.perf["description"],
            "expected_rtf": self.perf["rtf_factor"],
            "rtf_range": [self.perf["min_rtf"], self.perf["max_rtf"]],
            "supported_emotions": len(self.supported_emotions)
        }
    
    def cleanup(self):
        """清理资源"""
        logger.info("🧹 Fish Speech模拟器资源清理完成")
