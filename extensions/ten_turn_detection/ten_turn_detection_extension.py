#!/usr/bin/env python3
"""
TEN Turn Detection Extension for Stream-Omni
基于TEN Turn Detection实现智能说话轮换检测
支持finished、wait、unfinished三种状态判断
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any, List
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

class TenTurnDetectionExtension(Extension):
    """TEN Turn Detection扩展"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.turn_detector = None
        self.is_initialized = False
        
        # 文本缓冲和处理
        self.text_buffer = []
        self.max_buffer_size = 10
        self.processing_queue = queue.Queue()
        self.processing_thread = None
        self.stop_processing = threading.Event()
        
        # 状态管理
        self.current_turn_state = "unfinished"  # finished, wait, unfinished
        self.last_detection_time = None
        self.conversation_context = []
        
        # 配置参数
        self.confidence_threshold = 0.8
        self.context_window = 5  # 保留最近5轮对话作为上下文
        
        # 统计信息
        self.stats = {
            'total_detections': 0,
            'finished_count': 0,
            'wait_count': 0,
            'unfinished_count': 0,
            'processing_time': 0.0,
            'accuracy_feedback': []
        }
    
    def on_configure(self, ten_env: TenEnv) -> None:
        """配置扩展"""
        logger.info("🔄 配置TEN Turn Detection扩展")
        
        # 从配置中获取参数
        self.confidence_threshold = ten_env.get_property_float("confidence_threshold") or 0.8
        self.context_window = ten_env.get_property_int("context_window") or 5
        self.max_buffer_size = ten_env.get_property_int("max_buffer_size") or 10
        
        logger.info(f"📝 置信度阈值: {self.confidence_threshold}")
        logger.info(f"📝 上下文窗口: {self.context_window}")
        logger.info(f"📝 缓冲区大小: {self.max_buffer_size}")
        
        ten_env.on_configure_done()
    
    def on_init(self, ten_env: TenEnv) -> None:
        """初始化扩展"""
        logger.info("🚀 初始化TEN Turn Detection扩展")
        
        try:
            # 异步初始化Turn Detection引擎
            asyncio.create_task(self._init_turn_detector())
            ten_env.on_init_done()
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            ten_env.on_init_done()
    
    async def _init_turn_detector(self):
        """异步初始化Turn Detection引擎"""
        try:
            logger.info("📦 开始加载TEN Turn Detection引擎...")
            start_time = time.time()
            
            # 尝试加载官方TEN Turn Detection
            try:
                from ten_turn_detection_engine import TenTurnDetectionEngine
                self.turn_detector = TenTurnDetectionEngine(
                    confidence_threshold=self.confidence_threshold,
                    context_window=self.context_window
                )
                await self.turn_detector.initialize()
                
            except ImportError:
                logger.warning("⚠️ TEN Turn Detection库不可用，使用模拟引擎")
                from ten_turn_detection_simulator import TenTurnDetectionSimulator
                self.turn_detector = TenTurnDetectionSimulator(
                    confidence_threshold=self.confidence_threshold,
                    context_window=self.context_window
                )
                await self.turn_detector.initialize()
            
            load_time = time.time() - start_time
            self.is_initialized = True
            
            logger.info(f"✅ TEN Turn Detection引擎加载完成，耗时: {load_time:.2f}s")
            logger.info(f"🎯 特性: 上下文感知、多语言支持、高精度")
            
        except Exception as e:
            logger.error(f"❌ Turn Detection引擎加载失败: {e}")
            self.is_initialized = False
    
    def on_start(self, ten_env: TenEnv) -> None:
        """启动扩展"""
        logger.info("▶️ 启动TEN Turn Detection扩展")
        
        # 启动文本处理线程
        self.processing_thread = threading.Thread(target=self._text_processing_loop, daemon=True)
        self.processing_thread.start()
        
        ten_env.on_start_done()
    
    def on_stop(self, ten_env: TenEnv) -> None:
        """停止扩展"""
        logger.info("⏹️ 停止TEN Turn Detection扩展")
        
        # 停止处理
        self.stop_processing.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # 输出统计信息
        self._log_statistics()
        
        ten_env.on_stop_done()
    
    def on_deinit(self, ten_env: TenEnv) -> None:
        """反初始化扩展"""
        logger.info("🔚 反初始化TEN Turn Detection扩展")
        
        # 清理Turn Detection引擎
        if self.turn_detector:
            try:
                self.turn_detector.cleanup()
            except:
                pass
        
        ten_env.on_deinit_done()
    
    def on_cmd(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理命令"""
        cmd_name = cmd.get_name()
        
        if cmd_name == "detect_turn":
            self._handle_detect_turn_command(ten_env, cmd)
        elif cmd_name == "get_stats":
            self._handle_get_stats_command(ten_env, cmd)
        elif cmd_name == "set_threshold":
            self._handle_set_threshold_command(ten_env, cmd)
        elif cmd_name == "add_context":
            self._handle_add_context_command(ten_env, cmd)
        elif cmd_name == "clear_context":
            self._handle_clear_context_command(ten_env, cmd)
        else:
            logger.warning(f"⚠️ 未知命令: {cmd_name}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", f"Unknown command: {cmd_name}")
            ten_env.return_result(cmd_result, cmd)
    
    def on_data(self, ten_env: TenEnv, data: Data) -> None:
        """处理数据"""
        data_name = data.get_name()
        
        if data_name == "text_input":
            self._handle_text_input(ten_env, data)
        elif data_name == "asr_result":
            self._handle_asr_result(ten_env, data)
    
    def _handle_detect_turn_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理轮换检测命令"""
        try:
            if not self.is_initialized:
                logger.error("❌ Turn Detection引擎未初始化")
                cmd_result = CmdResult.create(StatusCode.ERROR)
                cmd_result.set_property_string("detail", "Turn detector not initialized")
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
            use_context = cmd.get_property_bool("use_context") or True
            
            logger.info(f"🔄 检测轮换: {text[:50]}...")
            
            # 异步处理检测
            asyncio.create_task(self._process_turn_detection_async(ten_env, cmd, text, use_context))
            
        except Exception as e:
            logger.error(f"❌ 轮换检测命令处理失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    async def _process_turn_detection_async(self, ten_env: TenEnv, cmd: Cmd, text: str, use_context: bool):
        """异步处理轮换检测"""
        try:
            start_time = time.time()
            
            # 准备上下文
            context = self.conversation_context if use_context else []
            
            # 执行轮换检测
            result = await self.turn_detector.detect_turn(text, context)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 更新统计
            self._update_stats(result, processing_time)
            
            # 更新状态
            self.current_turn_state = result['state']
            self.last_detection_time = end_time
            
            # 发送结果
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_string("turn_state", result['state'])
            cmd_result.set_property_float("confidence", result['confidence'])
            cmd_result.set_property_float("processing_time", processing_time)
            cmd_result.set_property_string("explanation", result.get('explanation', ''))
            
            ten_env.return_result(cmd_result, cmd)
            
            # 发送轮换事件
            self._send_turn_event(ten_env, result, text)
            
            logger.info(f"✅ 轮换检测完成: {result['state']} (置信度: {result['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"❌ 轮换检测处理失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_text_input(self, ten_env: TenEnv, data: Data):
        """处理文本输入"""
        try:
            text = data.get_property_string("text")
            if not text:
                return
            
            # 添加到处理队列
            self.processing_queue.put({
                'text': text,
                'timestamp': time.time(),
                'ten_env': ten_env,
                'type': 'text_input'
            })
            
        except Exception as e:
            logger.error(f"❌ 文本输入处理失败: {e}")
    
    def _handle_asr_result(self, ten_env: TenEnv, data: Data):
        """处理ASR结果"""
        try:
            text = data.get_property_string("text")
            is_final = data.get_property_bool("is_final") or False
            
            if not text:
                return
            
            # 只处理最终结果
            if is_final:
                self.processing_queue.put({
                    'text': text,
                    'timestamp': time.time(),
                    'ten_env': ten_env,
                    'type': 'asr_result'
                })
            
        except Exception as e:
            logger.error(f"❌ ASR结果处理失败: {e}")
    
    def _text_processing_loop(self):
        """文本处理循环"""
        logger.info("🔄 启动文本处理循环")
        
        while not self.stop_processing.is_set():
            try:
                # 获取文本数据
                try:
                    text_item = self.processing_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # 处理文本
                asyncio.run(self._process_text_item(text_item))
                
            except Exception as e:
                logger.error(f"❌ 文本处理循环错误: {e}")
                time.sleep(0.01)
        
        logger.info("🔄 文本处理循环结束")
    
    async def _process_text_item(self, text_item: Dict[str, Any]):
        """处理单个文本项"""
        try:
            text = text_item['text']
            ten_env = text_item['ten_env']
            
            # 执行轮换检测
            result = await self.turn_detector.detect_turn(text, self.conversation_context)
            
            # 更新状态
            self.current_turn_state = result['state']
            self.last_detection_time = text_item['timestamp']
            
            # 发送轮换事件
            self._send_turn_event(ten_env, result, text)
            
            # 添加到对话上下文
            self._add_to_context(text, result['state'])
            
        except Exception as e:
            logger.error(f"❌ 文本项处理失败: {e}")
    
    def _send_turn_event(self, ten_env: TenEnv, result: Dict[str, Any], text: str):
        """发送轮换事件"""
        try:
            data = Data.create("turn_detection_event")
            data.set_property_string("turn_state", result['state'])
            data.set_property_float("confidence", result['confidence'])
            data.set_property_string("text", text)
            data.set_property_float("timestamp", time.time())
            data.set_property_string("explanation", result.get('explanation', ''))
            
            ten_env.send_data(data)
            
        except Exception as e:
            logger.error(f"❌ 发送轮换事件失败: {e}")
    
    def _add_to_context(self, text: str, state: str):
        """添加到对话上下文"""
        context_item = {
            'text': text,
            'state': state,
            'timestamp': time.time()
        }
        
        self.conversation_context.append(context_item)
        
        # 保持上下文窗口大小
        if len(self.conversation_context) > self.context_window:
            self.conversation_context.pop(0)
    
    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """更新统计信息"""
        self.stats['total_detections'] += 1
        self.stats['processing_time'] += processing_time
        
        state = result['state']
        if state == 'finished':
            self.stats['finished_count'] += 1
        elif state == 'wait':
            self.stats['wait_count'] += 1
        elif state == 'unfinished':
            self.stats['unfinished_count'] += 1
    
    def _handle_get_stats_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理获取统计信息命令"""
        try:
            cmd_result = CmdResult.create(StatusCode.OK)
            
            # 基础统计
            cmd_result.set_property_int("total_detections", self.stats['total_detections'])
            cmd_result.set_property_int("finished_count", self.stats['finished_count'])
            cmd_result.set_property_int("wait_count", self.stats['wait_count'])
            cmd_result.set_property_int("unfinished_count", self.stats['unfinished_count'])
            
            # 性能统计
            if self.stats['total_detections'] > 0:
                avg_processing_time = self.stats['processing_time'] / self.stats['total_detections']
                cmd_result.set_property_float("avg_processing_time", avg_processing_time)
                
                # 状态分布
                total = self.stats['total_detections']
                cmd_result.set_property_float("finished_ratio", self.stats['finished_count'] / total)
                cmd_result.set_property_float("wait_ratio", self.stats['wait_count'] / total)
                cmd_result.set_property_float("unfinished_ratio", self.stats['unfinished_count'] / total)
            
            # 当前状态
            cmd_result.set_property_string("current_state", self.current_turn_state)
            cmd_result.set_property_float("confidence_threshold", self.confidence_threshold)
            cmd_result.set_property_int("context_size", len(self.conversation_context))
            
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
            
            old_threshold = self.confidence_threshold
            self.confidence_threshold = new_threshold
            
            # 更新检测器阈值
            if self.turn_detector:
                self.turn_detector.set_threshold(new_threshold)
            
            logger.info(f"🔧 置信度阈值更新: {old_threshold:.2f} -> {new_threshold:.2f}")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_float("old_threshold", old_threshold)
            cmd_result.set_property_float("new_threshold", new_threshold)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 设置阈值失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_add_context_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理添加上下文命令"""
        try:
            text = cmd.get_property_string("text")
            state = cmd.get_property_string("state") or "unfinished"
            
            if not text:
                raise ValueError("缺少文本参数")
            
            self._add_to_context(text, state)
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_int("context_size", len(self.conversation_context))
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 添加上下文失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _handle_clear_context_command(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理清除上下文命令"""
        try:
            old_size = len(self.conversation_context)
            self.conversation_context.clear()
            
            logger.info(f"🧹 清除对话上下文，原大小: {old_size}")
            
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_int("old_context_size", old_size)
            cmd_result.set_property_int("new_context_size", 0)
            ten_env.return_result(cmd_result, cmd)
            
        except Exception as e:
            logger.error(f"❌ 清除上下文失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("detail", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    def _log_statistics(self):
        """输出统计信息"""
        if self.stats['total_detections'] > 0:
            logger.info("📊 TEN Turn Detection统计:")
            logger.info(f"   总检测次数: {self.stats['total_detections']}")
            logger.info(f"   finished: {self.stats['finished_count']} ({self.stats['finished_count']/self.stats['total_detections']*100:.1f}%)")
            logger.info(f"   wait: {self.stats['wait_count']} ({self.stats['wait_count']/self.stats['total_detections']*100:.1f}%)")
            logger.info(f"   unfinished: {self.stats['unfinished_count']} ({self.stats['unfinished_count']/self.stats['total_detections']*100:.1f}%)")
            
            avg_time = self.stats['processing_time'] / self.stats['total_detections']
            logger.info(f"   平均处理时间: {avg_time*1000:.2f}ms")

def create_extension(name: str) -> Extension:
    """创建扩展实例"""
    return TenTurnDetectionExtension(name)
