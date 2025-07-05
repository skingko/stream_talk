#!/usr/bin/env python3
"""
Conversation Manager Extension for Stream-Omni
实现实时对话中断机制和全双工通信
支持自然的对话流和智能中断处理
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any, List
from enum import Enum
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

class ConversationState(Enum):
    """对话状态枚举"""
    IDLE = "idle"                    # 空闲状态
    LISTENING = "listening"          # 监听用户
    PROCESSING = "processing"        # 处理用户输入
    SPEAKING = "speaking"           # AI回复中
    INTERRUPTED = "interrupted"      # 被中断
    WAITING = "waiting"             # 等待用户继续

class ConversationManagerExtension(Extension):
    """对话管理器扩展"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.state = ConversationState.IDLE
        self.is_initialized = False
        
        # 组件引用
        self.vad_extension = None
        self.turn_detection_extension = None
        self.asr_extension = None
        self.llm_extension = None
        self.tts_extension = None
        
        # 对话管理
        self.current_conversation_id = None
        self.conversation_history = []
        self.pending_responses = queue.Queue()
        self.active_tts_task = None
        
        # 中断处理
        self.interruption_threshold = 0.3  # 300ms内检测到语音即中断
        self.last_user_speech_time = None
        self.interruption_count = 0
        
        # 配置参数
        self.max_silence_duration = 3.0    # 最大静音时长
        self.response_timeout = 10.0       # 响应超时时间
        self.interruption_sensitivity = 0.7 # 中断敏感度
        
        # 事件处理
        self.event_queue = queue.Queue()
        self.event_thread = None
        self.stop_events = threading.Event()
        
        # 统计信息
        self.stats = {
            'total_conversations': 0,
            'total_interruptions': 0,
            'successful_interruptions': 0,
            'average_response_time': 0.0,
            'state_transitions': {},
            'conversation_duration': 0.0
        }
    
    def on_configure(self, ten_env: TenEnv) -> None:
        """配置扩展"""
        logger.info("💬 配置对话管理器扩展")
        
        # 从配置中获取参数
        self.interruption_threshold = ten_env.get_property_float("interruption_threshold") or 0.3
        self.max_silence_duration = ten_env.get_property_float("max_silence_duration") or 3.0
        self.response_timeout = ten_env.get_property_float("response_timeout") or 10.0
        self.interruption_sensitivity = ten_env.get_property_float("interruption_sensitivity") or 0.7
        
        logger.info(f"📝 中断阈值: {self.interruption_threshold}s")
        logger.info(f"📝 最大静音: {self.max_silence_duration}s")
        logger.info(f"📝 响应超时: {self.response_timeout}s")
        logger.info(f"📝 中断敏感度: {self.interruption_sensitivity}")
        
        ten_env.on_configure_done()
    
    def on_init(self, ten_env: TenEnv) -> None:
        """初始化扩展"""
        logger.info("🚀 初始化对话管理器扩展")
        
        try:
            self.is_initialized = True
            ten_env.on_init_done()
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            ten_env.on_init_done()
    
    def on_start(self, ten_env: TenEnv) -> None:
        """启动扩展"""
        logger.info("▶️ 启动对话管理器扩展")
        
        # 启动事件处理线程
        self.event_thread = threading.Thread(target=self._event_processing_loop, daemon=True)
        self.event_thread.start()
        
        # 初始化对话状态
        self._transition_to_state(ConversationState.IDLE, ten_env)
        
        ten_env.on_start_done()
    
    def on_stop(self, ten_env: TenEnv) -> None:
        """停止扩展"""
        logger.info("⏹️ 停止对话管理器扩展")
        
        # 停止事件处理
        self.stop_events.set()
        
        if self.event_thread and self.event_thread.is_alive():
            self.event_thread.join(timeout=2.0)
        
        # 输出统计信息
        self._log_statistics()
        
        ten_env.on_stop_done()
    
    def on_deinit(self, ten_env: TenEnv) -> None:
        """反初始化扩展"""
        logger.info("🔚 反初始化对话管理器扩展")
        ten_env.on_deinit_done()
    
    def on_cmd(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理命令"""
        cmd_name = cmd.get_name()
        
        if cmd_name == "start_conversation":
            self._handle_start_conversation_command(ten_env, cmd)
        elif cmd_name == "stop_conversation":
            self._handle_stop_conversation_command(ten_env, cmd)
        elif cmd_name == "interrupt_response":
            self._handle_interrupt_response_command(ten_env, cmd)
        elif cmd_name == "get_state":
            self._handle_get_state_command(ten_env, cmd)
        elif cmd_name == "get_stats":
            self._handle_get_stats_command(ten_env, cmd)
        else:
            logger.warning(f"⚠️ 未知命令: {cmd_name}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", f"Unknown command: {cmd_name}")
            ten_env.return_result(cmd_result, cmd)
    
    def on_data(self, ten_env: TenEnv, data: Data) -> None:
        """处理数据"""
        data_name = data.get_name()
        
        if data_name == "vad_event":
            self._handle_vad_event(ten_env, data)
        elif data_name == "turn_detection_event":
            self._handle_turn_detection_event(ten_env, data)
        elif data_name == "asr_result":
            self._handle_asr_result(ten_env, data)
        elif data_name == "llm_response":
            self._handle_llm_response(ten_env, data)
        elif data_name == "tts_event":
            self._handle_tts_event(ten_env, data)
    
    def _handle_start_conversation_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理开始对话命令"""
        try:
            # 生成新的对话ID
            self.current_conversation_id = f"conv_{int(time.time())}"
            self.conversation_history.clear()
            self.stats['total_conversations'] += 1
            
            # 启动VAD监听
            self._send_command_to_extension("ten_vad", "start_listening", {}, ten_env)
            
            # 转换到监听状态
            self._transition_to_state(ConversationState.LISTENING, ten_env)
            
            logger.info(f"🎙️ 开始新对话: {self.current_conversation_id}")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_string("conversation_id", self.current_conversation_id)
            cmd_result.set_property_string("state", self.state.value)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 开始对话失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_stop_conversation_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理停止对话命令"""
        try:
            # 停止所有活动
            self._stop_all_activities(ten_env)
            
            # 转换到空闲状态
            self._transition_to_state(ConversationState.IDLE, ten_env)
            
            logger.info(f"🛑 停止对话: {self.current_conversation_id}")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_string("conversation_id", self.current_conversation_id)
            cmd_result.set_property_string("state", self.state.value)
            ten_env.return_result(cmd_result, cmd)
            
            self.current_conversation_id = None
            
        except Exception as e:
            logger.error(f"❌ 停止对话失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_vad_event(self, ten_env: TenEnv, data: Data):
        """处理VAD事件"""
        try:
            event_type = data.get_property_string("event_type")
            timestamp = data.get_property_float("timestamp")
            
            if event_type == "speech_start":
                self._on_speech_start(timestamp, ten_env)
            elif event_type == "speech_end":
                duration = data.get_property_float("duration")
                self._on_speech_end(timestamp, duration, ten_env)
                
        except Exception as e:
            logger.error(f"❌ VAD事件处理失败: {e}")
    
    def _handle_turn_detection_event(self, ten_env: TenEnv, data: Data):
        """处理轮换检测事件"""
        try:
            turn_state = data.get_property_string("turn_state")
            confidence = data.get_property_float("confidence")
            text = data.get_property_string("text")
            
            if turn_state == "finished" and confidence > self.interruption_sensitivity:
                # 用户说话结束，开始处理
                self._on_user_turn_finished(text, ten_env)
            elif turn_state == "wait":
                # 用户可能在等待，保持监听
                self._on_user_waiting(ten_env)
                
        except Exception as e:
            logger.error(f"❌ 轮换检测事件处理失败: {e}")
    
    def _handle_asr_result(self, ten_env: TenEnv, data: Data):
        """处理ASR结果"""
        try:
            text = data.get_property_string("text")
            is_final = data.get_property_bool("is_final")
            
            if is_final and text:
                # 最终识别结果，添加到对话历史
                self._add_to_conversation_history("user", text)
                
                # 发送给轮换检测
                self._send_data_to_extension("ten_turn_detection", "asr_result", {
                    "text": text,
                    "is_final": is_final
                }, ten_env)
                
        except Exception as e:
            logger.error(f"❌ ASR结果处理失败: {e}")
    
    def _handle_llm_response(self, ten_env: TenEnv, data: Data):
        """处理LLM响应"""
        try:
            response_text = data.get_property_string("text")
            is_final = data.get_property_bool("is_final")
            
            if response_text:
                if is_final:
                    # 最终响应，开始TTS
                    self._add_to_conversation_history("assistant", response_text)
                    self._start_tts_response(response_text, ten_env)
                else:
                    # 流式响应，可以开始流式TTS
                    self._stream_tts_response(response_text, ten_env)
                    
        except Exception as e:
            logger.error(f"❌ LLM响应处理失败: {e}")
    
    def _handle_tts_event(self, ten_env: TenEnv, data: Data):
        """处理TTS事件"""
        try:
            event_type = data.get_property_string("event_type")
            
            if event_type == "tts_start":
                self._transition_to_state(ConversationState.SPEAKING, ten_env)
            elif event_type == "tts_end":
                self._on_tts_finished(ten_env)
            elif event_type == "tts_interrupted":
                self._on_tts_interrupted(ten_env)
                
        except Exception as e:
            logger.error(f"❌ TTS事件处理失败: {e}")
    
    def _on_speech_start(self, timestamp: float, ten_env: TenEnv):
        """处理语音开始事件"""
        self.last_user_speech_time = timestamp
        
        # 检查是否需要中断
        if self.state == ConversationState.SPEAKING:
            # AI正在说话时检测到用户语音，执行中断
            time_since_speech = timestamp - (self.last_user_speech_time or 0)
            
            if time_since_speech <= self.interruption_threshold:
                self._interrupt_ai_response(ten_env)
        
        elif self.state == ConversationState.IDLE:
            # 空闲状态检测到语音，开始监听
            self._transition_to_state(ConversationState.LISTENING, ten_env)
    
    def _on_speech_end(self, timestamp: float, duration: float, ten_env: TenEnv):
        """处理语音结束事件"""
        if self.state == ConversationState.LISTENING:
            # 语音结束，等待轮换检测结果
            pass
    
    def _on_user_turn_finished(self, text: str, ten_env: TenEnv):
        """处理用户轮换结束"""
        if self.state == ConversationState.LISTENING:
            # 转换到处理状态
            self._transition_to_state(ConversationState.PROCESSING, ten_env)
            
            # 发送给LLM处理
            self._send_to_llm(text, ten_env)
    
    def _on_user_waiting(self, ten_env: TenEnv):
        """处理用户等待状态"""
        if self.state == ConversationState.PROCESSING:
            # 转换到等待状态
            self._transition_to_state(ConversationState.WAITING, ten_env)
    
    def _interrupt_ai_response(self, ten_env: TenEnv):
        """中断AI响应"""
        logger.info("⚡ 检测到用户中断，停止AI响应")
        
        # 停止TTS
        self._send_command_to_extension("fish_speech_tts", "stop", {}, ten_env)
        
        # 更新统计
        self.stats['total_interruptions'] += 1
        self.interruption_count += 1
        
        # 转换到被中断状态
        self._transition_to_state(ConversationState.INTERRUPTED, ten_env)
        
        # 快速转换到监听状态
        asyncio.create_task(self._delayed_transition_to_listening(ten_env))
    
    async def _delayed_transition_to_listening(self, ten_env: TenEnv):
        """延迟转换到监听状态"""
        await asyncio.sleep(0.1)  # 短暂延迟
        self._transition_to_state(ConversationState.LISTENING, ten_env)
    
    def _start_tts_response(self, text: str, ten_env: TenEnv):
        """开始TTS响应"""
        try:
            # 发送TTS命令
            self._send_command_to_extension("fish_speech_tts", "tts", {
                "text": text,
                "emotion": "neutral",
                "streaming": False
            }, ten_env)
            
        except Exception as e:
            logger.error(f"❌ 启动TTS失败: {e}")
    
    def _stream_tts_response(self, text: str, ten_env: TenEnv):
        """流式TTS响应"""
        try:
            # 发送流式TTS命令
            self._send_command_to_extension("fish_speech_tts", "tts", {
                "text": text,
                "emotion": "neutral",
                "streaming": True
            }, ten_env)
            
        except Exception as e:
            logger.error(f"❌ 流式TTS失败: {e}")
    
    def _on_tts_finished(self, ten_env: TenEnv):
        """TTS完成处理"""
        # 转换回监听状态，准备下一轮对话
        self._transition_to_state(ConversationState.LISTENING, ten_env)
    
    def _on_tts_interrupted(self, ten_env: TenEnv):
        """TTS被中断处理"""
        self.stats['successful_interruptions'] += 1
        # 已经在中断处理中转换状态
    
    def _send_to_llm(self, text: str, ten_env: TenEnv):
        """发送文本给LLM处理"""
        try:
            # 准备对话上下文
            context = self._prepare_llm_context()
            
            # 发送LLM命令
            self._send_command_to_extension("llm", "chat", {
                "text": text,
                "context": context,
                "streaming": True
            }, ten_env)
            
        except Exception as e:
            logger.error(f"❌ 发送LLM失败: {e}")
    
    def _prepare_llm_context(self) -> List[Dict[str, str]]:
        """准备LLM上下文"""
        # 返回最近的对话历史
        return self.conversation_history[-10:]  # 最近10轮对话
    
    def _add_to_conversation_history(self, role: str, content: str):
        """添加到对话历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
    
    def _transition_to_state(self, new_state: ConversationState, ten_env: TenEnv):
        """状态转换"""
        old_state = self.state
        self.state = new_state
        
        # 更新统计
        transition_key = f"{old_state.value}->{new_state.value}"
        self.stats['state_transitions'][transition_key] = self.stats['state_transitions'].get(transition_key, 0) + 1
        
        logger.info(f"🔄 状态转换: {old_state.value} -> {new_state.value}")
        
        # 发送状态变化事件
        self._send_state_change_event(old_state, new_state, ten_env)
    
    def _send_state_change_event(self, old_state: ConversationState, new_state: ConversationState, ten_env: TenEnv):
        """发送状态变化事件"""
        try:
            data = Data.create("conversation_state_change")
            data.set_property_string("old_state", old_state.value)
            data.set_property_string("new_state", new_state.value)
            data.set_property_float("timestamp", time.time())
            data.set_property_string("conversation_id", self.current_conversation_id or "")
            
            ten_env.send_data(data)
            
        except Exception as e:
            logger.error(f"❌ 发送状态变化事件失败: {e}")
    
    def _send_command_to_extension(self, extension_name: str, command: str, params: Dict[str, Any], ten_env: TenEnv):
        """发送命令给其他扩展"""
        try:
            cmd = Cmd.create(command)
            for key, value in params.items():
                if isinstance(value, str):
                    cmd.set_property_string(key, value)
                elif isinstance(value, bool):
                    cmd.set_property_bool(key, value)
                elif isinstance(value, (int, float)):
                    cmd.set_property_float(key, float(value))
            
            # 这里需要实际的扩展通信机制
            # ten_env.send_cmd_to_extension(extension_name, cmd)
            
        except Exception as e:
            logger.error(f"❌ 发送命令到{extension_name}失败: {e}")
    
    def _send_data_to_extension(self, extension_name: str, data_name: str, params: Dict[str, Any], ten_env: TenEnv):
        """发送数据给其他扩展"""
        try:
            data = Data.create(data_name)
            for key, value in params.items():
                if isinstance(value, str):
                    data.set_property_string(key, value)
                elif isinstance(value, bool):
                    data.set_property_bool(key, value)
                elif isinstance(value, (int, float)):
                    data.set_property_float(key, float(value))
            
            # 这里需要实际的扩展通信机制
            # ten_env.send_data_to_extension(extension_name, data)
            
        except Exception as e:
            logger.error(f"❌ 发送数据到{extension_name}失败: {e}")
    
    def _stop_all_activities(self, ten_env: TenEnv):
        """停止所有活动"""
        # 停止VAD
        self._send_command_to_extension("ten_vad", "stop_listening", {}, ten_env)
        
        # 停止TTS
        self._send_command_to_extension("fish_speech_tts", "stop", {}, ten_env)
        
        # 清空队列
        while not self.pending_responses.empty():
            try:
                self.pending_responses.get_nowait()
            except queue.Empty:
                break
    
    def _event_processing_loop(self):
        """事件处理循环"""
        logger.info("🔄 启动事件处理循环")
        
        while not self.stop_events.is_set():
            try:
                # 处理事件队列
                try:
                    event = self.event_queue.get(timeout=0.1)
                    # 处理事件
                    self._process_event(event)
                except queue.Empty:
                    continue
                    
            except Exception as e:
                logger.error(f"❌ 事件处理循环错误: {e}")
                time.sleep(0.01)
        
        logger.info("🔄 事件处理循环结束")
    
    def _process_event(self, event: Dict[str, Any]):
        """处理单个事件"""
        # 事件处理逻辑
        pass
    
    def _handle_get_state_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理获取状态命令"""
        try:
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_string("state", self.state.value)
            cmd_result.set_property_string("conversation_id", self.current_conversation_id or "")
            cmd_result.set_property_int("conversation_length", len(self.conversation_history))
            cmd_result.set_property_int("interruption_count", self.interruption_count)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 获取状态失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_get_stats_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理获取统计信息命令"""
        try:
            cmd_result = CmdResult.create(StatusCode.OK)
            
            # 基础统计
            cmd_result.set_property_int("total_conversations", self.stats['total_conversations'])
            cmd_result.set_property_int("total_interruptions", self.stats['total_interruptions'])
            cmd_result.set_property_int("successful_interruptions", self.stats['successful_interruptions'])
            
            # 中断成功率
            if self.stats['total_interruptions'] > 0:
                interruption_success_rate = self.stats['successful_interruptions'] / self.stats['total_interruptions']
                cmd_result.set_property_float("interruption_success_rate", interruption_success_rate)
            
            # 当前状态
            cmd_result.set_property_string("current_state", self.state.value)
            cmd_result.set_property_bool("is_in_conversation", self.current_conversation_id is not None)
            
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 获取统计信息失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_interrupt_response_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理中断响应命令"""
        try:
            # 手动触发中断
            self._interrupt_ai_response(ten_env)
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_string("result", "interrupted")
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 中断响应失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _log_statistics(self):
        """输出统计信息"""
        logger.info("📊 对话管理器统计:")
        logger.info(f"   总对话数: {self.stats['total_conversations']}")
        logger.info(f"   总中断数: {self.stats['total_interruptions']}")
        logger.info(f"   成功中断数: {self.stats['successful_interruptions']}")
        
        if self.stats['total_interruptions'] > 0:
            success_rate = self.stats['successful_interruptions'] / self.stats['total_interruptions'] * 100
            logger.info(f"   中断成功率: {success_rate:.1f}%")
        
        logger.info(f"   当前状态: {self.state.value}")

def create_extension(name: str) -> Extension:
    """创建扩展实例"""
    return ConversationManagerExtension(name)
