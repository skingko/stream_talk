#!/usr/bin/env python3
"""
TEN VAD Extension for Stream-Omni
基于TEN VAD实现低延迟、高性能的语音活动检测
支持持续监听和语音唤醒功能
"""

import asyncio
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, Callable
import threading
import queue

from ten import (
    Extension,
    TenEnv,
    Cmd,
    StatusCode,
    CmdResult,
    Data,
)

logger = logging.getLogger(__name__)

class TenVADExtension(Extension):
    """TEN VAD扩展"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.vad_engine = None
        self.is_initialized = False
        self.is_listening = False
        
        # 音频处理参数
        self.sample_rate = 16000  # TEN VAD要求16kHz
        self.frame_size = 160     # 10ms frame (16000 * 0.01)
        self.hop_size = 160       # 10ms hop
        
        # VAD参数
        self.vad_threshold = 0.5
        self.min_speech_duration = 0.3  # 最小语音持续时间
        self.min_silence_duration = 0.5  # 最小静音持续时间
        
        # 状态管理
        self.speech_state = "silence"  # silence, speech, uncertain
        self.speech_start_time = None
        self.silence_start_time = None
        self.last_speech_time = None
        
        # 音频缓冲
        self.audio_buffer = queue.Queue()
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'speech_frames': 0,
            'silence_frames': 0,
            'speech_events': 0,
            'silence_events': 0,
            'processing_time': 0.0
        }
    
    def on_configure(self, ten_env: TenEnv) -> None:
        """配置扩展"""
        logger.info("🔊 配置TEN VAD扩展")
        
        # 从配置中获取参数
        self.vad_threshold = ten_env.get_property_float("vad_threshold") or 0.5
        self.min_speech_duration = ten_env.get_property_float("min_speech_duration") or 0.3
        self.min_silence_duration = ten_env.get_property_float("min_silence_duration") or 0.5
        
        logger.info(f"📝 VAD阈值: {self.vad_threshold}")
        logger.info(f"📝 最小语音时长: {self.min_speech_duration}s")
        logger.info(f"📝 最小静音时长: {self.min_silence_duration}s")
        
        ten_env.on_configure_done()
    
    def on_init(self, ten_env: TenEnv) -> None:
        """初始化扩展"""
        logger.info("🚀 初始化TEN VAD扩展")
        
        try:
            # 异步初始化VAD引擎
            asyncio.create_task(self._init_vad_engine())
            ten_env.on_init_done()
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            ten_env.on_init_done()
    
    async def _init_vad_engine(self):
        """异步初始化VAD引擎"""
        try:
            logger.info("📦 开始加载TEN VAD引擎...")
            start_time = time.time()
            
            # 尝试加载TEN VAD
            try:
                from ten_vad_engine import TenVADEngine
                self.vad_engine = TenVADEngine(
                    sample_rate=self.sample_rate,
                    frame_size=self.frame_size,
                    threshold=self.vad_threshold
                )
                await self.vad_engine.initialize()
                
            except ImportError:
                logger.warning("⚠️ TEN VAD库不可用，使用模拟引擎")
                from ten_vad_simulator import TenVADSimulator
                self.vad_engine = TenVADSimulator(
                    sample_rate=self.sample_rate,
                    frame_size=self.frame_size,
                    threshold=self.vad_threshold
                )
                await self.vad_engine.initialize()
            
            load_time = time.time() - start_time
            self.is_initialized = True
            
            logger.info(f"✅ TEN VAD引擎加载完成，耗时: {load_time:.2f}s")
            logger.info(f"🎯 特性: 低延迟、高性能、轻量级")
            
        except Exception as e:
            logger.error(f"❌ VAD引擎加载失败: {e}")
            self.is_initialized = False
    
    def on_start(self, ten_env: TenEnv) -> None:
        """启动扩展"""
        logger.info("▶️ 启动TEN VAD扩展")
        
        # 启动音频处理线程
        self.processing_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self.processing_thread.start()
        
        ten_env.on_start_done()
    
    def on_stop(self, ten_env: TenEnv) -> None:
        """停止扩展"""
        logger.info("⏹️ 停止TEN VAD扩展")
        
        # 停止处理
        self.stop_processing.set()
        self.is_listening = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 输出统计信息
        self._log_statistics()
        
        ten_env.on_stop_done()
    
    def on_deinit(self, ten_env: TenEnv) -> None:
        """反初始化扩展"""
        logger.info("🔚 反初始化TEN VAD扩展")
        
        # 清理VAD引擎
        if self.vad_engine:
            try:
                self.vad_engine.cleanup()
            except:
                pass
        
        ten_env.on_deinit_done()
    
    def on_cmd(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理命令"""
        cmd_name = cmd.get_name()
        
        if cmd_name == "start_listening":
            self._handle_start_listening_command(ten_env, cmd)
        elif cmd_name == "stop_listening":
            self._handle_stop_listening_command(ten_env, cmd)
        elif cmd_name == "get_stats":
            self._handle_get_stats_command(ten_env, cmd)
        elif cmd_name == "set_threshold":
            self._handle_set_threshold_command(ten_env, cmd)
        else:
            logger.warning(f"⚠️ 未知命令: {cmd_name}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", f"Unknown command: {cmd_name}")
            ten_env.return_result(cmd_result, cmd)
    
    def on_data(self, ten_env: TenEnv, data: Data) -> None:
        """处理音频数据"""
        data_name = data.get_name()
        
        if data_name == "audio_frame" and self.is_listening:
            self._handle_audio_frame(ten_env, data)
    
    def _handle_start_listening_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理开始监听命令"""
        try:
            if not self.is_initialized:
                logger.error("❌ VAD引擎未初始化")
                cmd_result = CmdResult.create(StatusCode.ERROR)
                cmd_result.set_property_string("detail", "VAD engine not initialized")
                ten_env.return_result(cmd_result, cmd)
                return
            
            self.is_listening = True
            self.speech_state = "silence"
            self.speech_start_time = None
            self.silence_start_time = time.time()
            
            logger.info("🎧 开始语音监听...")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_bool("listening", True)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 开始监听失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_stop_listening_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理停止监听命令"""
        try:
            self.is_listening = False
            
            logger.info("🔇 停止语音监听")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_bool("listening", False)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 停止监听失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_audio_frame(self, ten_env: TenEnv, data: Data) -> None:
        """处理音频帧"""
        try:
            # 获取音频数据
            audio_data = data.get_property_buf("audio_data")
            if not audio_data:
                return
            
            # 转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # 添加到处理队列
            self.audio_buffer.put({
                'audio': audio_array,
                'timestamp': time.time(),
                'ten_env': ten_env
            })
            
        except Exception as e:
            logger.error(f"❌ 音频帧处理失败: {e}")
    
    def _audio_processing_loop(self):
        """音频处理循环"""
        logger.info("🔄 启动音频处理循环")
        
        while not self.stop_processing.is_set():
            try:
                # 获取音频数据
                try:
                    audio_item = self.audio_buffer.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if not self.is_listening:
                    continue
                
                # 处理音频帧
                self._process_audio_frame(
                    audio_item['audio'],
                    audio_item['timestamp'],
                    audio_item['ten_env']
                )
                
            except Exception as e:
                logger.error(f"❌ 音频处理循环错误: {e}")
                time.sleep(0.01)
        
        logger.info("🔄 音频处理循环结束")
    
    def _process_audio_frame(self, audio: np.ndarray, timestamp: float, ten_env: TenEnv):
        """处理单个音频帧"""
        try:
            start_time = time.time()
            
            # VAD检测
            is_speech = self.vad_engine.detect(audio)
            
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            self.stats['total_frames'] += 1
            
            # 更新统计
            if is_speech:
                self.stats['speech_frames'] += 1
            else:
                self.stats['silence_frames'] += 1
            
            # 状态机处理
            self._update_speech_state(is_speech, timestamp, ten_env)
            
        except Exception as e:
            logger.error(f"❌ 音频帧处理失败: {e}")
    
    def _update_speech_state(self, is_speech: bool, timestamp: float, ten_env: TenEnv):
        """更新语音状态"""
        current_state = self.speech_state
        
        if is_speech:
            if current_state == "silence":
                # 从静音转为语音
                if self.silence_start_time and (timestamp - self.silence_start_time) >= self.min_silence_duration:
                    self.speech_state = "speech"
                    self.speech_start_time = timestamp
                    self.silence_start_time = None
                    
                    # 发送语音开始事件
                    self._send_speech_event(ten_env, "speech_start", timestamp)
                    self.stats['speech_events'] += 1
                    
                    logger.info("🗣️ 检测到语音开始")
            
            # 更新最后语音时间
            self.last_speech_time = timestamp
        
        else:
            if current_state == "speech":
                # 从语音转为静音
                if self.speech_start_time and (timestamp - self.last_speech_time) >= self.min_speech_duration:
                    self.speech_state = "silence"
                    self.silence_start_time = timestamp
                    speech_duration = timestamp - self.speech_start_time
                    self.speech_start_time = None
                    
                    # 发送语音结束事件
                    self._send_speech_event(ten_env, "speech_end", timestamp, speech_duration)
                    self.stats['silence_events'] += 1
                    
                    logger.info(f"🔇 检测到语音结束，时长: {speech_duration:.2f}s")
    
    def _send_speech_event(self, ten_env: TenEnv, event_type: str, timestamp: float, duration: float = None):
        """发送语音事件"""
        try:
            data = Data.create("vad_event")
            data.set_property_string("event_type", event_type)
            data.set_property_float("timestamp", timestamp)
            data.set_property_string("state", self.speech_state)
            
            if duration is not None:
                data.set_property_float("duration", duration)
            
            ten_env.send_data(data)
            
        except Exception as e:
            logger.error(f"❌ 发送语音事件失败: {e}")
    
    def _handle_get_stats_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理获取统计信息命令"""
        try:
            cmd_result = CmdResult.create(StatusCode.OK)
            
            # 基础统计
            cmd_result.set_property_int("total_frames", self.stats['total_frames'])
            cmd_result.set_property_int("speech_frames", self.stats['speech_frames'])
            cmd_result.set_property_int("silence_frames", self.stats['silence_frames'])
            cmd_result.set_property_int("speech_events", self.stats['speech_events'])
            cmd_result.set_property_int("silence_events", self.stats['silence_events'])
            
            # 性能统计
            if self.stats['total_frames'] > 0:
                avg_processing_time = self.stats['processing_time'] / self.stats['total_frames']
                cmd_result.set_property_float("avg_processing_time", avg_processing_time)
                
                speech_ratio = self.stats['speech_frames'] / self.stats['total_frames']
                cmd_result.set_property_float("speech_ratio", speech_ratio)
            
            # 状态信息
            cmd_result.set_property_bool("is_listening", self.is_listening)
            cmd_result.set_property_string("current_state", self.speech_state)
            cmd_result.set_property_float("vad_threshold", self.vad_threshold)
            
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_set_threshold_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理设置阈值命令"""
        try:
            new_threshold = cmd.get_property_float("threshold")
            if new_threshold is None or not (0.0 <= new_threshold <= 1.0):
                raise ValueError("阈值必须在0.0-1.0之间")
            
            old_threshold = self.vad_threshold
            self.vad_threshold = new_threshold
            
            # 更新VAD引擎阈值
            if self.vad_engine:
                self.vad_engine.set_threshold(new_threshold)
            
            logger.info(f"🔧 VAD阈值更新: {old_threshold:.2f} -> {new_threshold:.2f}")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_float("old_threshold", old_threshold)
            cmd_result.set_property_float("new_threshold", new_threshold)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 设置阈值失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _log_statistics(self):
        """输出统计信息"""
        if self.stats['total_frames'] > 0:
            logger.info("📊 TEN VAD统计:")
            logger.info(f"   总帧数: {self.stats['total_frames']}")
            logger.info(f"   语音帧: {self.stats['speech_frames']} ({self.stats['speech_frames']/self.stats['total_frames']*100:.1f}%)")
            logger.info(f"   静音帧: {self.stats['silence_frames']} ({self.stats['silence_frames']/self.stats['total_frames']*100:.1f}%)")
            logger.info(f"   语音事件: {self.stats['speech_events']}")
            logger.info(f"   静音事件: {self.stats['silence_events']}")
            
            avg_time = self.stats['processing_time'] / self.stats['total_frames']
            logger.info(f"   平均处理时间: {avg_time*1000:.2f}ms/帧")

def create_extension(name: str) -> Extension:
    """创建扩展实例"""
    return TenVADExtension(name)
