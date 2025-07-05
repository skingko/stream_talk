#!/usr/bin/env python3
"""
Fish Speech TTS Extension for TEN Framework
基于Fish Speech实现高性能实时语音合成
"""

import asyncio
import logging
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator
import torch

from ten import (
    Extension,
    TenEnv,
    Cmd,
    StatusCode,
    CmdResult,
    Data,
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FishSpeechTTSExtension(Extension):
    """Fish Speech TTS扩展"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.tts_engine = None
        self.model_path = None
        self.device = None
        self.is_initialized = False
        
        # 性能统计
        self.stats = {
            'total_requests': 0,
            'total_processing_time': 0.0,
            'average_rtf': 0.0
        }
    
    def on_configure(self, ten_env: TenEnv) -> None:
        """配置扩展"""
        logger.info("🐟 配置Fish Speech TTS扩展")
        
        # 从配置中获取参数
        self.model_path = ten_env.get_property_string("model_path") or "models/fish-speech/openaudio-s1-mini"
        self.device = ten_env.get_property_string("device") or "auto"
        
        # 设置设备
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        logger.info(f"📝 模型路径: {self.model_path}")
        logger.info(f"🔧 使用设备: {self.device}")
        
        ten_env.on_configure_done()
    
    def on_init(self, ten_env: TenEnv) -> None:
        """初始化扩展"""
        logger.info("🚀 初始化Fish Speech TTS扩展")
        
        try:
            # 异步初始化模型
            asyncio.create_task(self._init_fish_speech_model())
            ten_env.on_init_done()
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            ten_env.on_init_done()
    
    async def _init_fish_speech_model(self):
        """异步初始化Fish Speech模型"""
        try:
            logger.info("📦 开始加载Fish Speech模型...")
            start_time = time.time()
            
            # 检查模型文件
            model_path = Path(self.model_path)
            if not model_path.exists():
                logger.error(f"❌ 模型路径不存在: {model_path}")
                return
            
            # 尝试加载官方Fish Speech API
            try:
                from fish_speech_tts_engine import FishSpeechEngine
                self.tts_engine = FishSpeechEngine(
                    model_path=str(model_path),
                    device=self.device
                )
                await self.tts_engine.initialize()
                
            except ImportError:
                logger.warning("⚠️ 官方Fish Speech API不可用，使用模拟引擎")
                from fish_speech_simulator import FishSpeechSimulator
                self.tts_engine = FishSpeechSimulator(
                    model_path=str(model_path),
                    device=self.device
                )
                await self.tts_engine.initialize()
            
            load_time = time.time() - start_time
            self.is_initialized = True
            
            logger.info(f"✅ Fish Speech模型加载完成，耗时: {load_time:.2f}s")
            logger.info(f"🎯 预期RTF: < 1.0 (实时性能)")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            self.is_initialized = False
    
    def on_start(self, ten_env: TenEnv) -> None:
        """启动扩展"""
        logger.info("▶️ 启动Fish Speech TTS扩展")
        ten_env.on_start_done()
    
    def on_stop(self, ten_env: TenEnv) -> None:
        """停止扩展"""
        logger.info("⏹️ 停止Fish Speech TTS扩展")
        
        # 清理资源
        if self.tts_engine:
            try:
                self.tts_engine.cleanup()
            except:
                pass
        
        # 输出统计信息
        if self.stats['total_requests'] > 0:
            logger.info("📊 Fish Speech TTS统计:")
            logger.info(f"   总请求数: {self.stats['total_requests']}")
            logger.info(f"   平均RTF: {self.stats['average_rtf']:.3f}")
            logger.info(f"   总处理时间: {self.stats['total_processing_time']:.2f}s")
        
        ten_env.on_stop_done()
    
    def on_deinit(self, ten_env: TenEnv) -> None:
        """反初始化扩展"""
        logger.info("🔚 反初始化Fish Speech TTS扩展")
        ten_env.on_deinit_done()
    
    def on_cmd(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理命令"""
        cmd_name = cmd.get_name()
        
        if cmd_name == "tts":
            self._handle_tts_command(ten_env, cmd)
        elif cmd_name == "get_stats":
            self._handle_get_stats_command(ten_env, cmd)
        else:
            logger.warning(f"⚠️ 未知命令: {cmd_name}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", f"Unknown command: {cmd_name}")
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_tts_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理TTS命令"""
        try:
            # 检查初始化状态
            if not self.is_initialized:
                logger.error("❌ Fish Speech模型未初始化")
                cmd_result = CmdResult.create(StatusCode.ERROR)
                cmd_result.set_property_string("detail", "Model not initialized")
                ten_env.return_result(cmd_result, cmd)
                return
            
            # 获取文本
            text = cmd.get_property_string("text")
            if not text:
                logger.error("❌ 缺少文本参数")
                cmd_result = CmdResult.create(StatusCode.ERROR)
                cmd_result.set_property_string("detail", "Missing text parameter")
                ten_env.return_result(cmd_result, cmd)
                return
            
            # 获取可选参数
            emotion = cmd.get_property_string("emotion") or "neutral"
            streaming = cmd.get_property_bool("streaming") or False
            
            logger.info(f"🎵 开始合成: {text[:50]}...")
            
            # 异步处理TTS
            asyncio.create_task(self._process_tts_async(ten_env, cmd, text, emotion, streaming))
            
        except Exception as e:
            logger.error(f"❌ TTS命令处理失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    async def _process_tts_async(self, ten_env: TenEnv, cmd: Cmd, text: str, emotion: str, streaming: bool) -> None:
        """异步处理TTS请求"""
        try:
            start_time = time.time()
            
            if streaming:
                # 流式处理
                async for audio_chunk in self._synthesize_streaming(text, emotion):
                    # 发送音频块
                    data = Data.create("audio_chunk")
                    data.set_property_buf("audio_data", audio_chunk.tobytes())
                    data.set_property_int("sample_rate", 22050)
                    data.set_property_string("format", "pcm_f32le")
                    ten_env.send_data(data)
                
                # 发送完成信号
                end_time = time.time()
                processing_time = end_time - start_time
                
                cmd_result = CmdResult.create(StatusCode.OK)
                cmd_result.set_property_float("processing_time", processing_time)
                cmd_result.set_property_bool("streaming", True)
                ten_env.return_result(cmd_result, cmd)
                
            else:
                # 批量处理
                audio_data = await self._synthesize_batch(text, emotion)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # 计算RTF
                estimated_duration = len(text) * 0.15  # 中文约每字符0.15秒
                rtf = processing_time / estimated_duration if estimated_duration > 0 else 0
                
                # 更新统计
                self._update_stats(processing_time, rtf)
                
                # 发送结果
                data = Data.create("audio_data")
                data.set_property_buf("audio_data", audio_data.tobytes())
                data.set_property_int("sample_rate", 22050)
                data.set_property_string("format", "pcm_f32le")
                data.set_property_float("duration", estimated_duration)
                ten_env.send_data(data)
                
                cmd_result = CmdResult.create(StatusCode.OK)
                cmd_result.set_property_float("processing_time", processing_time)
                cmd_result.set_property_float("rtf", rtf)
                cmd_result.set_property_bool("streaming", False)
                ten_env.return_result(cmd_result, cmd)
                
                logger.info(f"✅ 合成完成，RTF: {rtf:.3f}, 处理时间: {processing_time:.3f}s")
                
        except Exception as e:
            logger.error(f"❌ TTS处理失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    async def _synthesize_streaming(self, text: str, emotion: str) -> AsyncGenerator[np.ndarray, None]:
        """流式语音合成"""
        try:
            async for chunk in self.tts_engine.synthesize_streaming(text, emotion=emotion):
                yield chunk
        except Exception as e:
            logger.error(f"❌ 流式合成失败: {e}")
            raise
    
    async def _synthesize_batch(self, text: str, emotion: str) -> np.ndarray:
        """批量语音合成"""
        try:
            audio_data = await self.tts_engine.synthesize(text, emotion=emotion)
            return audio_data
        except Exception as e:
            logger.error(f"❌ 批量合成失败: {e}")
            raise
    
    def _handle_get_stats_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理获取统计信息命令"""
        try:
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_int("total_requests", self.stats['total_requests'])
            cmd_result.set_property_float("total_processing_time", self.stats['total_processing_time'])
            cmd_result.set_property_float("average_rtf", self.stats['average_rtf'])
            cmd_result.set_property_bool("is_initialized", self.is_initialized)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _update_stats(self, processing_time: float, rtf: float) -> None:
        """更新统计信息"""
        self.stats['total_requests'] += 1
        self.stats['total_processing_time'] += processing_time
        
        # 计算平均RTF
        if self.stats['total_requests'] > 0:
            self.stats['average_rtf'] = (
                (self.stats['average_rtf'] * (self.stats['total_requests'] - 1) + rtf) / 
                self.stats['total_requests']
            )

def create_extension(name: str) -> Extension:
    """创建扩展实例"""
    return FishSpeechTTSExtension(name)
