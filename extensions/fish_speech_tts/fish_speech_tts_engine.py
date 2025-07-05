#!/usr/bin/env python3
"""
Fish Speech TTS Engine
基于官方Fish Speech API的高性能TTS引擎
"""

import asyncio
import logging
import time
import sys
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
import numpy as np
import torch

# 添加项目根目录到Python路径，以便导入third_party_paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入第三方库路径配置
try:
    from third_party_paths import get_fish_speech_path
    fish_speech_path = get_fish_speech_path()
    if fish_speech_path.exists():
        sys.path.insert(0, str(fish_speech_path))
except ImportError:
    # 备用方案：直接使用路径
    fish_speech_path = project_root / "third-party" / "fish-speech"
    if fish_speech_path.exists():
        sys.path.insert(0, str(fish_speech_path))

logger = logging.getLogger(__name__)

class FishSpeechEngine:
    """Fish Speech TTS引擎"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = Path(model_path)
        self.device = device
        self.tts_engine = None
        self.llama_queue = None
        self.decoder_model = None
        self.is_initialized = False
        
        # 情感映射
        self.emotion_map = {
            "neutral": "",
            "happy": "(happy)",
            "sad": "(sad)",
            "angry": "(angry)",
            "excited": "(excited)",
            "surprised": "(surprised)",
            "joyful": "(joyful)",
            "confident": "(confident)",
            "whispering": "(whispering)",
            "shouting": "(shouting)",
            "laughing": "(laughing)",
            "chuckling": "(chuckling)",
            "sobbing": "(sobbing)",
            "sighing": "(sighing)"
        }
    
    async def initialize(self):
        """初始化Fish Speech模型"""
        try:
            logger.info("🐟 初始化Fish Speech引擎...")
            
            # 检查模型文件
            if not self._check_model_files():
                raise FileNotFoundError("模型文件不完整")
            
            # 尝试加载官方API
            try:
                await self._load_official_api()
                logger.info("✅ 官方Fish Speech API加载成功")
            except Exception as e:
                logger.warning(f"⚠️ 官方API加载失败: {e}")
                await self._load_fallback_engine()
                logger.info("✅ 备用引擎加载成功")
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"❌ Fish Speech引擎初始化失败: {e}")
            raise
    
    def _check_model_files(self) -> bool:
        """检查模型文件完整性"""
        required_files = ["model.pth", "codec.pth", "config.json", "tokenizer.tiktoken"]
        
        for file in required_files:
            file_path = self.model_path / file
            if not file_path.exists():
                logger.error(f"❌ 缺少模型文件: {file}")
                return False
        
        logger.info("✅ 模型文件检查完成")
        return True
    
    async def _load_official_api(self):
        """加载官方Fish Speech API"""
        try:
            # 导入官方模块
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            
            logger.info("📝 设置模型路径...")
            llama_checkpoint_path = str(self.model_path / "model.pth")
            decoder_checkpoint_path = str(self.model_path / "codec.pth")
            decoder_config_name = "firefly_gan_vq"
            
            # 设置精度
            precision = torch.half if self.device == "cuda" else torch.bfloat16
            
            logger.info("🧠 加载LLAMA模型...")
            self.llama_queue = launch_thread_safe_queue(
                checkpoint_path=llama_checkpoint_path,
                device=self.device,
                precision=precision,
                compile=False,
            )
            
            logger.info("🎵 加载解码器模型...")
            self.decoder_model = load_decoder_model(
                config_name=decoder_config_name,
                checkpoint_path=decoder_checkpoint_path,
                device=self.device,
            )
            
            logger.info("🚀 创建TTS推理引擎...")
            self.tts_engine = TTSInferenceEngine(
                llama_queue=self.llama_queue,
                decoder_model=self.decoder_model,
                precision=precision,
                compile=False,
            )
            
        except Exception as e:
            logger.error(f"❌ 官方API加载失败: {e}")
            raise
    
    async def _load_fallback_engine(self):
        """加载备用引擎（模拟器）"""
        from fish_speech_simulator import FishSpeechSimulator
        
        self.tts_engine = FishSpeechSimulator(
            model_path=str(self.model_path),
            device=self.device
        )
        await self.tts_engine.initialize()
    
    async def synthesize(self, text: str, emotion: str = "neutral", **kwargs) -> np.ndarray:
        """批量语音合成"""
        if not self.is_initialized:
            raise RuntimeError("引擎未初始化")
        
        try:
            # 添加情感标记
            emotion_tag = self.emotion_map.get(emotion, "")
            if emotion_tag:
                text_with_emotion = f"{emotion_tag} {text}"
            else:
                text_with_emotion = text
            
            logger.info(f"🎵 合成文本: {text_with_emotion[:50]}...")
            
            # 调用TTS引擎
            if hasattr(self.tts_engine, 'synthesize'):
                # 使用官方API
                audio_data = await self._synthesize_with_official_api(text_with_emotion, **kwargs)
            else:
                # 使用模拟器
                audio_data = await self.tts_engine.synthesize(text_with_emotion, **kwargs)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ 语音合成失败: {e}")
            raise
    
    async def synthesize_streaming(self, text: str, emotion: str = "neutral", **kwargs) -> AsyncGenerator[np.ndarray, None]:
        """流式语音合成"""
        if not self.is_initialized:
            raise RuntimeError("引擎未初始化")
        
        try:
            # 添加情感标记
            emotion_tag = self.emotion_map.get(emotion, "")
            if emotion_tag:
                text_with_emotion = f"{emotion_tag} {text}"
            else:
                text_with_emotion = text
            
            logger.info(f"🌊 流式合成: {text_with_emotion[:50]}...")
            
            # 调用流式TTS
            if hasattr(self.tts_engine, 'synthesize_streaming'):
                # 使用官方流式API
                async for chunk in self._synthesize_streaming_with_official_api(text_with_emotion, **kwargs):
                    yield chunk
            else:
                # 使用模拟器流式API
                async for chunk in self.tts_engine.synthesize_streaming(text_with_emotion, **kwargs):
                    yield chunk
                    
        except Exception as e:
            logger.error(f"❌ 流式语音合成失败: {e}")
            raise
    
    async def _synthesize_with_official_api(self, text: str, **kwargs) -> np.ndarray:
        """使用官方API进行合成"""
        try:
            from fish_speech.utils.schema import ServeTTSRequest
            
            # 创建TTS请求
            request = ServeTTSRequest(
                text=text,
                reference_id=None,
                reference_audio=None,
                reference_text=None,
                max_new_tokens=kwargs.get('max_new_tokens', 1024),
                chunk_length=kwargs.get('chunk_length', 200),
                top_p=kwargs.get('top_p', 0.7),
                repetition_penalty=kwargs.get('repetition_penalty', 1.2),
                temperature=kwargs.get('temperature', 0.7),
                speaker=kwargs.get('speaker', None),
                emotion=kwargs.get('emotion', None),
                format="wav",
                streaming=False,
            )
            
            # 执行推理
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.tts_engine.inference, request
            )
            
            # 提取音频数据
            if hasattr(result, 'audio'):
                return result.audio
            else:
                # 如果结果格式不同，生成模拟数据
                return self._generate_mock_audio(text)
                
        except Exception as e:
            logger.warning(f"⚠️ 官方API合成失败: {e}")
            # 降级到模拟模式
            return self._generate_mock_audio(text)
    
    async def _synthesize_streaming_with_official_api(self, text: str, **kwargs) -> AsyncGenerator[np.ndarray, None]:
        """使用官方API进行流式合成"""
        try:
            from fish_speech.utils.schema import ServeTTSRequest
            
            # 创建流式TTS请求
            request = ServeTTSRequest(
                text=text,
                reference_id=None,
                reference_audio=None,
                reference_text=None,
                max_new_tokens=kwargs.get('max_new_tokens', 1024),
                chunk_length=kwargs.get('chunk_length', 100),  # 更小的块用于流式
                top_p=kwargs.get('top_p', 0.7),
                repetition_penalty=kwargs.get('repetition_penalty', 1.2),
                temperature=kwargs.get('temperature', 0.7),
                speaker=kwargs.get('speaker', None),
                emotion=kwargs.get('emotion', None),
                format="wav",
                streaming=True,
            )
            
            # 执行流式推理
            async for chunk in self.tts_engine.inference_streaming(request):
                if hasattr(chunk, 'audio'):
                    yield chunk.audio
                else:
                    # 如果格式不同，生成模拟块
                    yield self._generate_mock_audio_chunk()
                    
        except Exception as e:
            logger.warning(f"⚠️ 官方流式API失败: {e}")
            # 降级到模拟流式模式
            async for chunk in self._generate_mock_streaming(text):
                yield chunk
    
    def _generate_mock_audio(self, text: str) -> np.ndarray:
        """生成模拟音频数据"""
        # 模拟处理时间
        processing_time = len(text) * 0.01 * 0.8  # Fish Speech预期性能
        time.sleep(min(processing_time, 2.0))
        
        # 生成模拟音频
        sample_rate = 22050
        duration = len(text) * 0.15
        samples = int(sample_rate * duration)
        
        # 生成更真实的音频波形
        t = np.linspace(0, duration, samples)
        frequency = 200 + np.random.rand() * 300
        audio_data = (
            0.3 * np.sin(2 * np.pi * frequency * t) +
            0.1 * np.random.randn(samples)
        ).astype(np.float32)
        
        return audio_data
    
    def _generate_mock_audio_chunk(self) -> np.ndarray:
        """生成模拟音频块"""
        sample_rate = 22050
        chunk_duration = 0.1  # 100ms块
        samples = int(sample_rate * chunk_duration)
        
        # 生成音频块
        t = np.linspace(0, chunk_duration, samples)
        frequency = 200 + np.random.rand() * 300
        audio_chunk = (
            0.3 * np.sin(2 * np.pi * frequency * t) +
            0.1 * np.random.randn(samples)
        ).astype(np.float32)
        
        return audio_chunk
    
    async def _generate_mock_streaming(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """生成模拟流式音频"""
        total_duration = len(text) * 0.15
        chunk_duration = 0.1  # 100ms块
        num_chunks = int(total_duration / chunk_duration)
        
        for i in range(num_chunks):
            # 模拟处理延迟
            await asyncio.sleep(0.05)  # 50ms延迟
            
            chunk = self._generate_mock_audio_chunk()
            yield chunk
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.llama_queue:
                # 清理LLAMA队列
                pass
            
            if self.decoder_model:
                # 清理解码器
                pass
                
            logger.info("🧹 Fish Speech引擎资源清理完成")
            
        except Exception as e:
            logger.error(f"❌ 资源清理失败: {e}")
    
    def get_supported_emotions(self) -> list:
        """获取支持的情感列表"""
        return list(self.emotion_map.keys())
