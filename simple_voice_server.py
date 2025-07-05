#!/usr/bin/env python3
"""
简化的实时语音交互服务器
专注于解决WebSocket连接稳定性问题
"""

import asyncio
import logging
import json
import time
import base64
import numpy as np
import os
import sys
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass

# FastAPI和WebSocket
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests

@dataclass
class VADConfig:
    """VAD配置类 - 参考TEN VAD配置"""
    energy_threshold: float = 0.005  # 能量阈值，降低避免过敏感
    peak_threshold: float = 0.02     # 峰值阈值
    silence_duration_frames: int = 30  # 静音持续帧数 (1.5秒)
    min_speech_frames: int = 16       # 最小语音帧数 (0.8秒)
    min_audio_length: int = 800       # 最小音频长度 (50ms)
    prefix_padding_ms: int = 120      # 前缀填充毫秒数
    hop_size_ms: int = 50            # 跳跃大小毫秒数

@dataclass
class WhisperConfig:
    """Whisper配置类 - 使用large-v3-turbo优化中文支持"""
    model_size: str = "large-v3-turbo"  # 使用large-v3-turbo模型，中文支持最佳
    device: str = "cpu"               # 设备
    compute_type: str = "int8"        # 计算类型
    beam_size: int = 5                # beam搜索大小，large模型可以用更大的beam_size
    temperature: float = 0.0          # 温度参数
    compression_ratio_threshold: float = 2.4  # 压缩比阈值
    log_prob_threshold: float = -1.0  # 对数概率阈值
    no_speech_threshold: float = 0.6  # 无语音阈值

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleVoiceServer:
    """简化的语音交互服务器"""

    def __init__(self):
        self.app = FastAPI(title="Simple Voice Interaction Server")
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_states: Dict[str, Dict[str, Any]] = {}

        # 语音缓冲区 - 用于累积语音数据
        self.speech_buffers: Dict[str, Dict[str, Any]] = {}

        # 回音抑制 - 记录AI说话时间
        self.ai_speaking_periods: Dict[str, Dict[str, float]] = {}  # connection_id -> {"start_time": float, "end_time": float}
        self.echo_suppression_delay = 2.0  # AI说话结束后2秒内忽略用户输入

        # LLM配置
        self.lm_studio_url = "http://localhost:1234/v1/chat/completions"

        # Spark-TTS引擎
        self.spark_tts = None

        # 音频块缓冲和去重
        self.audio_chunk_buffer = {}  # 按connection_id缓冲音频块
        self.sent_audio_hashes = {}   # 记录已发送的音频块哈希，防止重复

        # 配置实例 - 参考TEN框架优化
        self.vad_config = VADConfig()
        self.whisper_config = WhisperConfig()

        self._setup_routes()
        self._setup_middleware()

    async def initialize_spark_tts(self):
        """初始化Spark-TTS"""
        try:
            logger.info("⚡ 初始化Spark-TTS...")

            from spark_tts_wrapper import SparkTTSWrapper

            self.spark_tts = SparkTTSWrapper()

            # 检查模型是否加载成功
            if self.spark_tts.is_model_loaded():
                logger.info("✅ Spark-TTS模型加载成功")
                model_info = self.spark_tts.get_model_info()
                logger.info(f"📋 模型信息: {model_info}")
                return True
            else:
                raise RuntimeError("Spark-TTS模型未完全加载")

        except Exception as e:
            logger.error(f"❌ Spark-TTS初始化失败: {e}")
            raise RuntimeError(f"Spark-TTS初始化失败: {e}")
    
    def _setup_middleware(self):
        """设置中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Simple Voice Interaction Server",
                "version": "1.0.0",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "active_connections": len(self.active_connections),
                "conversations": len(self.conversation_states)
            }
        
        @self.app.websocket("/ws/voice")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket_connection(websocket)

        @self.app.post("/api/chat")
        async def chat_endpoint(request: dict):
            return await self._handle_chat_request(request)
    
    async def _handle_websocket_connection(self, websocket: WebSocket):
        """处理WebSocket连接"""
        connection_id = f"conn_{int(time.time() * 1000)}_{id(websocket)}"
        
        try:
            await websocket.accept()
            self.active_connections[connection_id] = websocket
            logger.info(f"🔌 WebSocket连接建立: {connection_id}")
            
            # 发送连接确认
            await self._send_to_connection(connection_id, {
                "type": "connection_established",
                "data": {
                    "connection_id": connection_id,
                    "server_time": time.time()
                }
            })
            
            # 消息处理循环 - 实现健壮的循环监听逻辑
            while connection_id in self.active_connections:
                try:
                    message = await websocket.receive_text()
                    await self._process_websocket_message(connection_id, message)
                except WebSocketDisconnect:
                    logger.info(f"🔌 WebSocket正常断开: {connection_id}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ 消息处理异常: {e}, 继续监听...")
                    # 发生异常时，确保系统能恢复到监听状态
                    await self._ensure_listening_state(connection_id)
                    # 短暂延迟后继续监听
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"❌ WebSocket连接错误: {e}")
        
        finally:
            # 清理连接
            await self._cleanup_connection(connection_id)
    
    async def _process_websocket_message(self, connection_id: str, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            logger.debug(f"📨 收到消息: {message_type} from {connection_id}")
            
            if message_type == "ping":
                # 心跳响应
                await self._send_to_connection(connection_id, {
                    "type": "pong",
                    "data": {"timestamp": time.time()}
                })
            
            elif message_type == "start_conversation":
                # 开始对话
                conversation_id = data.get("data", {}).get("conversation_id", f"conv_{int(time.time() * 1000)}")
                await self._start_conversation(connection_id, conversation_id)
            
            elif message_type == "stop_conversation":
                # 停止对话
                await self._stop_conversation(connection_id)
            
            elif message_type == "text_message":
                # 处理文本消息
                await self._process_text_message(connection_id, data.get("data", {}))
            
            elif message_type == "init":
                # 初始化消息
                logger.info(f"🔧 收到初始化消息: {connection_id}")
                await self._send_to_connection(connection_id, {
                    "type": "init_response",
                    "data": {
                        "status": "ready",
                        "connection_id": connection_id,
                        "timestamp": time.time()
                    }
                })

            elif message_type == "audio_data":
                # 处理音频数据
                logger.info(f"🎵 开始处理音频数据: {connection_id}")
                await self._process_audio_data(connection_id, data.get("data", {}))

            else:
                logger.warning(f"⚠️ 未知消息类型: {message_type}")
        
        except Exception as e:
            logger.error(f"❌ 消息处理失败: {connection_id}, {e}")
    
    async def _start_conversation(self, connection_id: str, conversation_id: str):
        """开始对话"""
        try:
            self.conversation_states[connection_id] = {
                "conversation_id": conversation_id,
                "started_at": time.time(),
                "message_count": 0
            }
            
            await self._send_to_connection(connection_id, {
                "type": "conversation_started",
                "data": {"conversation_id": conversation_id}
            })
            
            logger.info(f"🎯 对话开始: {conversation_id}")
        
        except Exception as e:
            logger.error(f"❌ 开始对话失败: {e}")
    
    async def _stop_conversation(self, connection_id: str):
        """停止对话"""
        try:
            conversation_state = self.conversation_states.get(connection_id, {})
            conversation_id = conversation_state.get("conversation_id", "unknown")
            
            await self._send_to_connection(connection_id, {
                "type": "conversation_stopped",
                "data": {"conversation_id": conversation_id}
            })
            
            if connection_id in self.conversation_states:
                del self.conversation_states[connection_id]
            
            logger.info(f"🛑 对话停止: {conversation_id}")
        
        except Exception as e:
            logger.error(f"❌ 停止对话失败: {e}")
    
    async def _process_text_message(self, connection_id: str, text_data: Dict[str, Any]):
        """处理文本消息"""
        try:
            text = text_data.get("text", "")
            if not text:
                return
            
            conversation_state = self.conversation_states.get(connection_id, {})
            conversation_id = conversation_state.get("conversation_id", "unknown")
            
            logger.info(f"💬 收到文本消息: '{text}' from {conversation_id}")
            
            # 发送转录结果确认
            await self._send_to_connection(connection_id, {
                "type": "transcription_result",
                "data": {
                    "text": text,
                    "confidence": 1.0,
                    "conversation_id": conversation_id
                }
            })
            
            # 获取LLM响应
            response = await self._get_llm_response(text)
            
            # 发送LLM响应
            await self._send_to_connection(connection_id, {
                "type": "llm_response",
                "data": {
                    "text": response,
                    "conversation_id": conversation_id
                }
            })
            
            # 模拟TTS事件
            await self._send_to_connection(connection_id, {
                "type": "tts_start",
                "data": {
                    "text": response,
                    "conversation_id": conversation_id
                }
            })
            
            # 短暂延迟模拟TTS播放
            await asyncio.sleep(1)
            
            await self._send_to_connection(connection_id, {
                "type": "tts_end",
                "data": {
                    "text": response,
                    "conversation_id": conversation_id
                }
            })
            
            # 更新消息计数
            if connection_id in self.conversation_states:
                self.conversation_states[connection_id]["message_count"] += 1
        
        except Exception as e:
            logger.error(f"❌ 文本处理失败: {e}")
    
    async def _get_llm_response(self, text: str) -> str:
        """获取LLM响应"""
        try:
            logger.info(f"🤖 请求LLM响应: '{text}'")
            
            # 检查LM Studio是否可用
            try:
                response = requests.post(
                    self.lm_studio_url,
                    json={
                        "model": "qwen2.5-32b-instruct",
                        "messages": [
                            {"role": "system", "content": """你是一个友好的AI助手。请用简洁、自然的中文回答用户的问题。

重要：你的回答将通过语音合成播放，请在回答中适当使用情感和语调标记来增强表达效果：

基础情感标记：
(angry) (sad) (excited) (surprised) (satisfied) (delighted) (scared) (worried) (upset) (nervous) (frustrated) (depressed) (empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed) (grateful) (confident) (interested) (curious) (confused) (joyful)

高级情感标记：
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) (impatient) (guilty) (scornful) (panicked) (furious) (reluctant) (keen) (disapproving) (negative) (denying) (astonished) (serious) (sarcastic) (conciliative) (comforting) (sincere) (sneering) (hesitating) (yielding) (painful) (awkward) (amused)

语调标记：
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)

特殊音效：
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting) (groaning) (crowd laughing) (background laughter) (audience laughing)

使用示例：
- 表达高兴：(joyful)很高兴为您服务！
- 表达关心：(empathetic)我理解您的感受。
- 表达自信：(confident)我可以帮您解决这个问题。
- 表达温柔：(soft tone)让我来为您详细解释一下。

请根据对话内容和情境，自然地在回答中加入合适的情感标记，让语音更生动有趣。"""},
                            {"role": "user", "content": text}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 1000,  # 增加token限制
                        "stream": False
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    llm_response = data["choices"][0]["message"]["content"].strip()

                    # 过滤掉思考过程（<think>标签内容）
                    original_response = llm_response
                    if "</think>" in llm_response:
                        # 有完整的</think>标签，取标签后的内容
                        llm_response = llm_response.split("</think>")[-1].strip()
                    elif "<think>" in llm_response:
                        # 如果有<think>但没有</think>，说明响应被截断
                        # 检查是否有<think>之前的内容
                        before_think = llm_response.split("<think>")[0].strip()
                        if before_think:
                            llm_response = before_think
                        else:
                            # 如果<think>之前没有内容，尝试从原始响应中提取有用信息
                            # 或者使用默认回复
                            llm_response = "我理解了您的意思。"

                    # 如果过滤后为空，提供默认回复
                    if not llm_response:
                        llm_response = "我理解了您的意思。"

                    logger.info(f"✅ LLM原始响应: '{data['choices'][0]['message']['content'][:100]}...'")
                    logger.info(f"✅ LLM过滤后响应: '{llm_response}'")
                    return llm_response
                else:
                    logger.error(f"❌ LLM请求失败: {response.status_code}")
                    return "抱歉，我现在无法回答您的问题。"
            
            except requests.exceptions.ConnectionError:
                logger.error("❌ LM Studio未连接")
                raise Exception("LM Studio未连接")

            except requests.exceptions.Timeout:
                logger.error("❌ LLM请求超时")
                raise Exception("LLM请求超时")
        
        except Exception as e:
            logger.error(f"❌ LLM请求异常: {e}")
            raise
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """发送消息到连接"""
        try:
            websocket = self.active_connections.get(connection_id)
            if websocket:
                await websocket.send_json(message)
                logger.debug(f"📤 消息已发送: {message.get('type')} to {connection_id}")
                return True
            else:
                logger.warning(f"⚠️ 连接不存在: {connection_id}")
                return False
        except Exception as e:
            # 检查是否是连接断开相关的错误
            error_msg = str(e).lower()
            if "not connected" in error_msg or "connection closed" in error_msg:
                logger.debug(f"🔌 连接已断开: {connection_id}")
            else:
                logger.error(f"❌ 发送消息失败: {connection_id}, {e}")

            # 当发送失败时，说明连接已经断开，主动清理连接状态
            if connection_id in self.active_connections:
                logger.info(f"🧹 检测到连接断开，清理连接状态: {connection_id}")
                del self.active_connections[connection_id]
            return False

    def _is_ai_speaking_or_recently_spoke(self, connection_id: str) -> bool:
        """检查AI是否正在说话或刚刚说完话（回音抑制）"""
        if connection_id not in self.ai_speaking_periods:
            return False

        speaking_info = self.ai_speaking_periods[connection_id]
        current_time = time.time()

        # 如果AI正在说话（还没有结束时间）
        if speaking_info["end_time"] is None:
            return True

        # 如果AI刚说完话，在抑制延迟时间内
        time_since_end = current_time - speaking_info["end_time"]
        if time_since_end < self.echo_suppression_delay:
            return True

        return False

    async def _ensure_listening_state(self, connection_id: str):
        """确保系统恢复到监听状态"""
        try:
            if connection_id in self.active_connections:
                logger.info(f"🔄 恢复监听状态: {connection_id}")
                await self._send_to_connection(connection_id, {
                    "type": "listening_ready",
                    "data": {
                        "message": "系统已恢复，准备接收语音输入",
                        "timestamp": time.time()
                    }
                })
        except Exception as e:
            logger.warning(f"⚠️ 恢复监听状态失败: {e}")

    async def _cleanup_connection(self, connection_id: str):
        """清理连接"""
        try:
            # 移除连接
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            # 移除对话状态
            if connection_id in self.conversation_states:
                del self.conversation_states[connection_id]

            # 清理音频相关数据
            if connection_id in self.audio_chunk_buffer:
                del self.audio_chunk_buffer[connection_id]

            if connection_id in self.sent_audio_hashes:
                del self.sent_audio_hashes[connection_id]

            # 清理语音缓冲区
            conversation_id = f"conv_{connection_id}"
            if conversation_id in self.speech_buffers:
                del self.speech_buffers[conversation_id]

            # 清理AI说话记录
            if connection_id in self.ai_speaking_periods:
                del self.ai_speaking_periods[connection_id]

            logger.info(f"🧹 连接清理完成: {connection_id}")

        except Exception as e:
            logger.error(f"❌ 连接清理失败: {e}")

    async def _handle_chat_request(self, request: dict):
        """处理文本聊天请求"""
        try:
            message = request.get("message", "")
            if not message:
                return {"error": "消息内容不能为空"}

            # 获取LLM响应
            response = await self._get_llm_response(message)

            return {
                "response": response,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"❌ 聊天请求处理失败: {e}")
            return {"error": str(e)}

    def _reset_speech_buffer(self, conversation_id: str):
        """重置语音缓冲区"""
        self.speech_buffers[conversation_id] = {
            "buffer": [],
            "last_speech_time": time.time(),
            "silence_count": 0,
            "is_speaking": False,
            "speech_started": False,
            "current_transcript": "",
            "last_transcription_time": 0
        }
        logger.info(f"🔄 重置语音缓冲区: {conversation_id}")

    async def _process_audio_data(self, connection_id: str, audio_data: Dict[str, Any]):
        """处理音频数据 - 使用VAD累积和Turn Detection判断"""
        try:
            # 获取音频数据
            audio_b64 = audio_data.get("audio", "")
            if not audio_b64:
                logger.warning(f"⚠️ 空音频数据: {connection_id}")
                return

            conversation_id = audio_data.get("conversation_id", connection_id)

            # 初始化语音缓冲区
            if conversation_id not in self.speech_buffers:
                self._reset_speech_buffer(conversation_id)

            buffer_info = self.speech_buffers[conversation_id]

            # 解码音频数据
            try:
                audio_bytes = base64.b64decode(audio_b64)
                # 转换为numpy数组 (int16 -> float32)
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

                logger.debug(f"🎤 音频解码: 长度={len(audio_array)}, 范围=[{audio_array.min():.3f}, {audio_array.max():.3f}]")

                # VAD检测 - 检查是否有语音活动
                has_speech = await self._vad_detect_speech(audio_array)

                if has_speech:
                    # 检测到语音，累积到缓冲区
                    buffer_info["buffer"].extend(audio_array.tolist())
                    buffer_info["last_speech_time"] = time.time()
                    buffer_info["silence_count"] = 0

                    if not buffer_info["speech_started"]:
                        buffer_info["speech_started"] = True
                        buffer_info["is_speaking"] = True
                        logger.info(f"🎤 开始语音段: {conversation_id}")

                        # 发送语音开始事件
                        await self._send_to_connection(connection_id, {
                            "type": "speech_start",
                            "data": {
                                "conversation_id": conversation_id,
                                "timestamp": time.time()
                            }
                        })
                else:
                    # 没有检测到语音
                    if buffer_info["speech_started"]:
                        buffer_info["silence_count"] += 1

                        # 使用配置的静音检测参数
                        if buffer_info["silence_count"] >= self.vad_config.silence_duration_frames:
                            await self._finalize_speech_segment(connection_id, conversation_id)

            except Exception as decode_error:
                logger.error(f"❌ 音频解码失败: {decode_error}")
                return

        except Exception as e:
            logger.error(f"❌ 音频处理失败: {e}")

    async def _vad_detect_speech(self, audio_array: np.ndarray) -> bool:
        """VAD语音活动检测 - 使用配置化参数"""
        try:
            # 使用配置的参数
            energy = np.sqrt(np.mean(audio_array ** 2))

            # 添加音频长度检查，避免噪音误触发
            if len(audio_array) < self.vad_config.min_audio_length:
                return False

            # 添加峰值检测，避免持续低能量噪音
            peak_energy = np.max(np.abs(audio_array))

            # 综合判断：能量阈值 + 峰值检测
            has_speech = (energy > self.vad_config.energy_threshold) and (peak_energy > self.vad_config.peak_threshold)

            if has_speech:
                logger.debug(f"🎤 VAD检测到语音: 能量={energy:.6f}, 峰值={peak_energy:.6f}")

            return has_speech

        except Exception as e:
            logger.error(f"❌ VAD检测失败: {e}")
            return False

    async def _finalize_speech_segment(self, connection_id: str, conversation_id: str):
        """完成语音段处理"""
        try:
            if conversation_id not in self.speech_buffers:
                return

            buffer_info = self.speech_buffers[conversation_id]

            # 防止重复处理
            if not buffer_info["speech_started"]:
                return

            # 回音抑制：检查AI是否正在说话或刚说完话
            if self._is_ai_speaking_or_recently_spoke(connection_id):
                logger.info(f"🔇 回音抑制：AI正在说话或刚说完话，忽略用户语音: {conversation_id}")
                # 重置状态但不清空buffer，继续监听
                buffer_info["speech_started"] = False
                buffer_info["silence_count"] = 0
                buffer_info["buffer"].clear()  # 清空缓冲区避免累积
                return

            # 使用配置的最小音频长度检测
            min_audio_samples = self.vad_config.min_speech_frames * 1000  # 转换为样本数 (16kHz)
            if len(buffer_info["buffer"]) < min_audio_samples:
                logger.info(f"🎤 语音段太短，忽略: {conversation_id} (长度: {len(buffer_info['buffer'])}, 需要: {min_audio_samples})")
                # 重置状态但不清空buffer，继续监听
                buffer_info["speech_started"] = False
                buffer_info["silence_count"] = 0
                return

            logger.info(f"🎤 完成语音段，开始转录: {conversation_id} (音频长度: {len(buffer_info['buffer'])})")

            # 标记为正在处理
            buffer_info["speech_started"] = False
            buffer_info["is_speaking"] = False

            # 发送语音结束事件
            await self._send_to_connection(connection_id, {
                "type": "speech_end",
                "data": {
                    "conversation_id": conversation_id,
                    "timestamp": time.time()
                }
            })

            # 完整转录
            audio_array = np.array(buffer_info["buffer"], dtype=np.float32)
            transcript = await self._transcribe_with_whisper(audio_array)

            logger.info(f"🎤 转录结果: '{transcript}'")

            # 发送转录结果
            await self._send_to_connection(connection_id, {
                "type": "transcription_result",
                "data": {
                    "text": transcript,
                    "confidence": 0.9,
                    "is_final": True,
                    "conversation_id": conversation_id
                }
            })

            # 使用Turn Detection判断是否应该回复
            if transcript.strip():
                should_respond = await self._check_turn_completion(transcript, conversation_id)

                if should_respond:
                    # 获取LLM响应
                    response = await self._get_llm_response(transcript)

                    # 发送LLM响应
                    await self._send_to_connection(connection_id, {
                        "type": "llm_response",
                        "data": {
                            "text": response,
                            "conversation_id": conversation_id,
                            "timestamp": time.time()
                        }
                    })

                    # 使用Spark-TTS播放
                    await self._speak_with_spark_tts(connection_id, response, conversation_id)
                else:
                    logger.info("🎤 Turn Detection: 用户还想继续说话，保持监听")

            # 清空缓冲区，准备下一段语音
            buffer_info["buffer"] = []

        except Exception as e:
            logger.error(f"❌ 语音段处理失败: {e}")

    async def _check_turn_completion(self, transcript: str, conversation_id: str) -> bool:
        """使用Turn Detection检查轮次是否完成"""
        try:
            # 简化的Turn Detection逻辑
            # 在真实实现中，这里应该调用TEN Turn Detection

            # 简单规则：如果句子以句号、问号、感叹号结尾，认为完成
            if transcript.strip().endswith(('。', '？', '！', '.', '?', '!')):
                logger.info("🎯 Turn Detection: 检测到句子结束，应该回复")
                return True

            # 如果句子较短且完整，也认为完成
            if len(transcript.strip()) > 3 and len(transcript.strip()) < 50:
                logger.info("🎯 Turn Detection: 短句完整，应该回复")
                return True

            logger.info("🎯 Turn Detection: 用户可能还想继续说话")
            return False

        except Exception as e:
            logger.error(f"❌ Turn Detection失败: {e}")
            # 降级处理：默认认为完成
            return True

    async def _speak_with_spark_tts(self, connection_id: str, text: str, conversation_id: str):
        """使用Spark-TTS播放文本"""
        try:
            # 记录AI开始说话时间（回音抑制）
            current_time = time.time()
            self.ai_speaking_periods[connection_id] = {
                "start_time": current_time,
                "end_time": None
            }

            # 发送TTS开始事件
            await self._send_to_connection(connection_id, {
                "type": "tts_start",
                "data": {
                    "text": text,
                    "conversation_id": conversation_id,
                    "timestamp": current_time
                }
            })

            if self.spark_tts and self.spark_tts.is_model_loaded():
                # 使用Spark-TTS流式合成
                logger.info(f"⚡ 开始Spark-TTS播放: {text}")
                chunk_count = 0

                # 生成音频流 - 使用标准语速（Spark-TTS支持的速度值）
                for audio_chunk in self.spark_tts.synthesize_stream(text, speed=1.0):
                    chunk_count += 1

                    # 提取音频数据
                    audio_array = audio_chunk.get("audio", np.array([]))
                    sample_rate = audio_chunk.get("sample_rate", 22050)

                    if len(audio_array) == 0:
                        continue

                    # 转换为int16格式的WAV字节数据
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    wav_bytes = audio_int16.tobytes()

                    # 音频块去重检查
                    import hashlib
                    audio_hash = hashlib.md5(wav_bytes).hexdigest()

                    if connection_id not in self.sent_audio_hashes:
                        self.sent_audio_hashes[connection_id] = set()

                    if audio_hash in self.sent_audio_hashes[connection_id]:
                        logger.debug(f"🔄 跳过重复音频块: {chunk_count}")
                        continue

                    self.sent_audio_hashes[connection_id].add(audio_hash)

                    # 将音频数据编码为base64
                    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

                    # 发送音频块
                    success = await self._send_to_connection(connection_id, {
                        "type": "audio_chunk",
                        "data": {
                            "audio": audio_b64,
                            "chunk_id": chunk_count,
                            "sample_rate": sample_rate,
                            "conversation_id": conversation_id,
                            "timestamp": time.time()
                        }
                    })

                    if not success:
                        logger.warning(f"⚠️ 发送音频块失败，停止TTS播放: {chunk_count}")
                        break

                    logger.debug(f"🎵 发送音频块 {chunk_count}: {len(wav_bytes)} 字节")

                    # 添加小延迟，防止音频块发送过快
                    import asyncio
                    await asyncio.sleep(0.05)  # 50ms延迟

                logger.info(f"✅ Fish Audio TTS播放完成，共{chunk_count}个音频块")

            else:
                # 降级到模拟TTS
                logger.warning("⚠️ Fish Audio TTS不可用，使用模拟TTS")
                import asyncio
                await asyncio.sleep(len(text) * 0.1)  # 模拟播放时间

            # 无论如何都要尝试恢复到监听状态
            try:
                # 记录AI说话结束时间（回音抑制）
                current_time = time.time()
                if connection_id in self.ai_speaking_periods:
                    self.ai_speaking_periods[connection_id]["end_time"] = current_time

                # 发送TTS结束事件
                if connection_id in self.active_connections:
                    await self._send_to_connection(connection_id, {
                        "type": "tts_end",
                        "data": {
                            "text": text,
                            "conversation_id": conversation_id,
                            "timestamp": current_time
                        }
                    })

                # 确保恢复到监听状态
                await self._ensure_listening_state(connection_id)

            except Exception as e:
                logger.warning(f"⚠️ TTS完成后处理异常: {e}")
                # 即使发生异常，也要尝试恢复监听状态
                try:
                    await self._ensure_listening_state(connection_id)
                except:
                    logger.error(f"❌ 无法恢复监听状态: {connection_id}")

        except Exception as e:
            logger.error(f"❌ Fish Audio TTS播放失败: {e}")

            # 发送TTS错误事件
            await self._send_to_connection(connection_id, {
                "type": "tts_error",
                "data": {
                    "text": text,
                    "error": str(e),
                    "conversation_id": conversation_id,
                    "timestamp": time.time()
                }
            })

    async def _transcribe_with_whisper(self, audio_array: np.ndarray) -> str:
        """使用Whisper进行语音转录"""
        try:
            # 检查是否有Whisper模型
            if not hasattr(self, 'whisper_model') or self.whisper_model is None:
                # 初始化Whisper模型 - 使用配置化参数
                try:
                    from faster_whisper import WhisperModel
                    logger.info("🎯 初始化Whisper模型...")
                    # 使用配置的参数
                    self.whisper_model = WhisperModel(
                        self.whisper_config.model_size,
                        device=self.whisper_config.device,
                        compute_type=self.whisper_config.compute_type,
                        num_workers=1,  # 单线程避免资源竞争
                        download_root="models/whisper"  # 本地模型目录
                    )
                    logger.info("✅ Whisper模型初始化成功")
                except ImportError:
                    logger.error("❌ faster-whisper未安装")
                    return "语音识别不可用"
                except Exception as e:
                    logger.error(f"❌ Whisper模型初始化失败: {e}")
                    return "语音识别初始化失败"

            # 检查音频数据
            if len(audio_array) < 1600:  # 至少0.1秒的音频
                logger.warning(f"⚠️ 音频数据太短: {len(audio_array)} 样本")
                return ""

            # 检查音频能量
            audio_energy = np.sqrt(np.mean(audio_array ** 2))
            logger.debug(f"🎤 音频能量: {audio_energy:.6f}")
            if audio_energy < 0.001:
                logger.warning(f"⚠️ 音频能量太低: {audio_energy:.6f}")
                return ""

            # 创建临时WAV文件
            import tempfile
            import wave

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                # 写入WAV文件
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16位
                    wav_file.setframerate(16000)  # 16kHz
                    # 转换为int16
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())

                logger.debug(f"🎤 开始Whisper转录，文件: {temp_file.name}")

                # 使用Whisper large-v3-turbo转录 - 专门优化中文识别
                segments, _ = self.whisper_model.transcribe(
                    temp_file.name,
                    language="zh",  # 中文
                    beam_size=self.whisper_config.beam_size,
                    initial_prompt="以下是普通话的句子，包含日常对话内容。",  # 优化中文提示
                    vad_filter=False,  # 禁用内置VAD，使用我们自己的VAD
                    word_timestamps=False,  # 禁用词级时间戳提高速度
                    condition_on_previous_text=False,  # 禁用上下文依赖提高准确性
                    temperature=self.whisper_config.temperature,
                    compression_ratio_threshold=self.whisper_config.compression_ratio_threshold,
                    log_prob_threshold=self.whisper_config.log_prob_threshold,
                    no_speech_threshold=self.whisper_config.no_speech_threshold,
                    # large-v3-turbo专用优化参数
                    repetition_penalty=1.1,  # 重复惩罚，避免重复识别
                    length_penalty=1.0,      # 长度惩罚
                    patience=1               # 耐心参数，提高准确性
                )

                # 提取转录文本
                transcript = ""
                for segment in segments:
                    logger.debug(f"🎤 Whisper片段: '{segment.text}'")
                    transcript += segment.text

                # 清理临时文件
                import os
                os.unlink(temp_file.name)

                return transcript.strip()

        except Exception as e:
            logger.error(f"❌ Whisper转录失败: {e}")
            return "转录失败"

# 全局服务器实例
voice_server = SimpleVoiceServer()

@voice_server.app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 初始化简化语音交互服务器...")

    # 初始化Spark-TTS引擎
    await voice_server.initialize_spark_tts()
    logger.info("✅ Spark-TTS引擎初始化成功")

    logger.info("✅ 简化语音交互服务器启动完成")

@voice_server.app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 简化语音交互服务器关闭")

def main():
    """主函数"""
    logger.info("🚀 启动简化语音交互服务器...")
    
    uvicorn.run(
        "simple_voice_server:voice_server.app",
        host="0.0.0.0",
        port=8002,  # 使用新端口
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
