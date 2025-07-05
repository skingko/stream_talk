"""
LM Studio LLM扩展
连接LM Studio + Qwen3-30B-A3模型，支持流式响应和思考内容过滤
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional
import aiohttp
from ten import (
    Extension,
    TenEnv,
    Cmd,
    StatusCode,
    CmdResult,
    Data,
)

logger = logging.getLogger(__name__)


class LMStudioLLMExtension(Extension):
    """LM Studio LLM扩展"""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.is_initialized = False
        
        # LM Studio配置
        self.lm_studio_url = "http://localhost:1234/v1/chat/completions"
        self.model_name = "qwen3-30b-a3b"  # 根据实际模型名称调整
        self.max_tokens = 2048
        self.temperature = 0.7
        self.top_p = 0.9
        
        # 会话管理
        self.session = None
        self.conversation_history = []
        
        # 系统提示
        self.system_prompt = """你是一个智能语音助手，名叫小助手。请用自然、友好的语调回复用户。
特点：
1. 回复要简洁明了，适合语音交互
2. 语调要亲切自然，像朋友聊天一样
3. 避免过长的回复，控制在50字以内
4. 可以适当使用语气词，让对话更生动
5. 支持中文对话"""
    
    async def on_init(self, ten_env: TenEnv) -> None:
        """初始化扩展"""
        try:
            logger.info("🚀 初始化LM Studio LLM扩展...")
            
            # 创建HTTP会话
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # 测试连接
            await self._test_connection()
            
            self.is_initialized = True
            logger.info("✅ LM Studio LLM扩展初始化成功")
            
        except Exception as e:
            logger.error(f"❌ LM Studio LLM扩展初始化失败: {e}")
            self.is_initialized = False
    
    async def on_deinit(self, ten_env: TenEnv) -> None:
        """清理扩展"""
        if self.session:
            await self.session.close()
        logger.info("🛑 LM Studio LLM扩展已清理")
    
    async def on_cmd(self, ten_env: TenEnv, cmd: Cmd) -> None:
        """处理命令"""
        cmd_name = cmd.get_name()
        
        if cmd_name == "chat":
            await self._handle_chat_command(ten_env, cmd)
        elif cmd_name == "reset_conversation":
            await self._handle_reset_command(ten_env, cmd)
        else:
            logger.warning(f"未知命令: {cmd_name}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("error", f"未知命令: {cmd_name}")
            ten_env.return_result(cmd_result, cmd)
    
    async def _test_connection(self):
        """测试LM Studio连接"""
        try:
            test_payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": "测试连接"}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            async with self.session.post(
                self.lm_studio_url,
                json=test_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    logger.info("✅ LM Studio连接测试成功")
                else:
                    raise Exception(f"连接测试失败，状态码: {response.status}")
                    
        except Exception as e:
            raise Exception(f"LM Studio连接失败: {e}")
    
    async def _handle_chat_command(self, ten_env: TenEnv, cmd: Cmd):
        """处理聊天命令"""
        try:
            # 获取参数
            user_text = cmd.get_property_string("text")
            context = cmd.get_property_from_json("context") or []
            streaming = cmd.get_property_bool("streaming") or True
            
            if not user_text:
                raise ValueError("缺少用户输入文本")
            
            logger.info(f"🤖 处理LLM请求: {user_text}")
            
            # 准备消息
            messages = self._prepare_messages(user_text, context)
            
            # 发送请求
            if streaming:
                await self._handle_streaming_chat(ten_env, cmd, messages, user_text)
            else:
                await self._handle_non_streaming_chat(ten_env, cmd, messages, user_text)
                
        except Exception as e:
            logger.error(f"❌ 处理聊天命令失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("error", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    async def _handle_streaming_chat(self, ten_env: TenEnv, cmd: Cmd, messages: List[Dict], user_text: str):
        """处理流式聊天"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": True
            }

            full_response = ""
            thinking_content = ""
            is_thinking = False
            output_content = ""

            async with self.session.post(
                self.lm_studio_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:

                if response.status != 200:
                    raise Exception(f"LM Studio请求失败，状态码: {response.status}")

                async for line in response.content:
                    line = line.decode('utf-8').strip()

                    if line.startswith('data: '):
                        data_str = line[6:]  # 移除 'data: ' 前缀

                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')

                                if content:
                                    full_response += content

                                    # 处理思考内容过滤
                                    filtered_content = self._filter_thinking_content(content, thinking_content, is_thinking)

                                    if filtered_content['has_output']:
                                        output_content += filtered_content['output']

                                        # 发送流式响应给TTS（只发送实际输出内容）
                                        await self._send_llm_response(ten_env, filtered_content['output'], False)

                                    # 更新状态
                                    thinking_content = filtered_content['thinking']
                                    is_thinking = filtered_content['is_thinking']

                        except json.JSONDecodeError:
                            continue

            # 发送最终响应
            if output_content:
                await self._send_llm_response(ten_env, output_content, True)

            # 更新对话历史（只保存输出内容）
            self._update_conversation_history(user_text, output_content or "我需要思考一下...")

            # 返回成功结果
            cmd_result = CmdResult.create(StatusCode.OK)
            cmd_result.set_property_string("response", output_content)
            ten_env.return_result(cmd_result, cmd)

        except Exception as e:
            logger.error(f"❌ 流式聊天失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("error", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    async def _handle_non_streaming_chat(self, ten_env: TenEnv, cmd: Cmd, messages: List[Dict], user_text: str):
        """处理非流式聊天"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stream": False
            }
            
            async with self.session.post(
                self.lm_studio_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    raise Exception(f"LM Studio请求失败，状态码: {response.status}")
                
                result = await response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    ai_response = result['choices'][0]['message']['content']
                    
                    # 发送响应
                    await self._send_llm_response(ten_env, ai_response, True)
                    
                    # 更新对话历史
                    self._update_conversation_history(user_text, ai_response)
                    
                    # 返回成功结果
                    cmd_result = CmdResult.create(StatusCode.OK)
                    cmd_result.set_property_string("response", ai_response)
                    ten_env.return_result(cmd_result, cmd)
                else:
                    raise Exception("LM Studio返回格式错误")
                    
        except Exception as e:
            logger.error(f"❌ 非流式聊天失败: {e}")
            cmd_result = CmdResult.create(StatusCode.ERROR)
            cmd_result.set_property_string("error", str(e))
            ten_env.return_result(cmd_result, cmd)
    
    async def _send_llm_response(self, ten_env: TenEnv, text: str, is_final: bool):
        """发送LLM响应"""
        data = Data.create("llm_response")
        data.set_property_string("text", text)
        data.set_property_bool("is_final", is_final)
        data.set_property_int("timestamp", int(time.time() * 1000))
        ten_env.send_data(data)
    
    def _prepare_messages(self, user_text: str, context: List[Dict]) -> List[Dict]:
        """准备消息列表"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # 添加上下文
        if context:
            messages.extend(context[-10:])  # 最近10轮对话
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_text})
        
        return messages
    
    def _filter_thinking_content(self, content: str, current_thinking: str, is_thinking: bool) -> Dict[str, Any]:
        """过滤思考内容，只返回实际输出"""
        result = {
            'thinking': current_thinking,
            'is_thinking': is_thinking,
            'output': '',
            'has_output': False
        }

        # 检查是否开始思考
        if '<think>' in content:
            result['is_thinking'] = True
            # 提取思考开始前的内容
            parts = content.split('<think>')
            if parts[0]:
                result['output'] = parts[0]
                result['has_output'] = True
            # 保存思考内容
            if len(parts) > 1:
                result['thinking'] = current_thinking + parts[1]

        # 检查是否结束思考
        elif '</think>' in content:
            if is_thinking:
                parts = content.split('</think>')
                # 添加到思考内容
                result['thinking'] = current_thinking + parts[0]
                result['is_thinking'] = False
                # 提取思考结束后的内容
                if len(parts) > 1 and parts[1]:
                    result['output'] = parts[1]
                    result['has_output'] = True
            else:
                # 不在思考状态，直接输出
                result['output'] = content
                result['has_output'] = True

        # 正在思考中
        elif is_thinking:
            result['thinking'] = current_thinking + content
            result['is_thinking'] = True

        # 正常输出
        else:
            result['output'] = content
            result['has_output'] = True

        return result

    def _update_conversation_history(self, user_text: str, ai_response: str):
        """更新对话历史"""
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": ai_response})

        # 保持历史长度
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    async def _handle_reset_command(self, ten_env: TenEnv, cmd: Cmd):
        """处理重置对话命令"""
        self.conversation_history.clear()
        logger.info("🔄 对话历史已重置")
        
        cmd_result = CmdResult.create(StatusCode.OK)
        cmd_result.set_property_string("message", "对话历史已重置")
        ten_env.return_result(cmd_result, cmd)
