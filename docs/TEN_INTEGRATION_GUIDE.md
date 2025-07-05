# Stream-Omni TEN框架集成指南

## 🎯 概述

Stream-Omni已成功集成TEN框架，实现了真正的实时语音对话功能。本指南将帮助您了解和使用这个强大的多模态AI系统。

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vue3 前端     │    │  FastAPI 后端   │    │   TEN 框架      │
│                 │    │                 │    │                 │
│ • 语音界面      │◄──►│ • WebSocket API │◄──►│ • VAD 检测      │
│ • 实时转录      │    │ • 流式处理      │    │ • Turn Detection│
│ • 状态管理      │    │ • 模型集成      │    │ • 音频处理      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 1. 启动系统

```bash
# 使用一键启动脚本
./start_stream_omni.sh

# 或手动启动
# 后端
source venv/bin/activate
python stream_omni_api.py

# 前端
cd frontend
npm run dev
```

### 2. 访问界面

- **前端界面**: http://localhost:5173
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

### 3. 测试系统

```bash
# 运行完整测试
python test_stream_omni.py
```

## 🎤 功能特性

### TEN框架集成

- ✅ **TEN VAD**: 实时语音活动检测
- ✅ **TEN Turn Detection**: 智能说话人轮换检测
- ✅ **实时音频处理**: 256样本帧处理
- ✅ **唤醒词检测**: 支持"小助手"、"你好"、"唤醒"

### 语音对话功能

- 🎯 **实时VAD**: 检测语音活动，能量阈值自适应
- 📝 **语音转录**: 实时ASR转录（可集成Whisper）
- 🤖 **智能回复**: LLM生成自然语言回复
- 🔄 **状态管理**: 监听→处理→回复→监听循环
- ⚡ **低延迟**: WebSocket实时通信

### 前端界面

- 🎨 **现代UI**: Vue3 + Element Plus响应式设计
- 🔊 **可视化**: 实时语音波形和状态指示
- 🎛️ **控制面板**: 音量控制、模式切换
- 📊 **状态显示**: 实时转录、对话历史
- 🔧 **测试模式**: TEN组件测试界面

## 📋 使用指南

### 语音对话模式

1. **切换到语音模式**: 点击侧边栏的🎤图标
2. **开始对话**: 点击"开始语音对话"按钮
3. **授权麦克风**: 允许浏览器访问麦克风
4. **开始说话**: 系统会自动检测语音并转录
5. **获得回复**: AI会自动生成并"播放"回复
6. **继续对话**: 系统自动回到监听状态

### TEN测试模式

1. **切换到测试模式**: 点击侧边栏的🔧图标
2. **连接测试**: 测试后端连接和WebSocket
3. **TEN组件测试**: 验证VAD和Turn Detection
4. **语音测试**: 录制音频并查看VAD结果
5. **对话测试**: 发送文本消息测试LLM

### 文本对话模式

1. **切换到文本模式**: 点击侧边栏的💬图标
2. **输入消息**: 在输入框中输入文本
3. **发送消息**: 按Enter或点击发送按钮
4. **查看回复**: AI会生成文本回复

## 🔧 技术细节

### TEN组件配置

```python
# VAD配置
vad_config = {
    "sample_rate": 16000,
    "frame_size": 256,
    "energy_threshold": 0.01,
    "wake_words": ["小助手", "你好", "唤醒"]
}

# Turn Detection配置
turn_config = {
    "silence_threshold": 1.0,  # 1秒静音检测
    "speech_threshold": 0.3,   # 语音检测阈值
    "turn_timeout": 5.0        # 轮换超时
}
```

### WebSocket消息格式

```javascript
// 音频数据
{
  "type": "audio_data",
  "data": {
    "audio": "base64_encoded_audio",
    "sample_rate": 16000,
    "conversation_id": "conv_123"
  }
}

// VAD结果
{
  "type": "vad_result",
  "data": {
    "speech_detected": true,
    "energy": 0.5,
    "timestamp": 1234567890
  }
}

// ASR结果
{
  "type": "asr_result",
  "data": {
    "transcript": "用户说的话",
    "confidence": 0.85,
    "is_final": true
  }
}

// LLM回复
{
  "type": "llm_response",
  "data": {
    "response": "AI的回复",
    "conversation_id": "conv_123"
  }
}
```

### API端点

```
GET  /                     # 系统信息
GET  /health              # 健康检查
POST /api/chat            # 聊天接口
POST /api/voice/process   # 语音处理
WS   /ws/voice           # WebSocket语音通信
```

## 🛠️ 开发指南

### 添加新的TEN组件

1. 在`agents/ten_packages/extension/`创建新扩展
2. 实现TEN扩展接口
3. 创建Python包装器
4. 在`stream_omni_api.py`中集成
5. 更新前端处理逻辑

### 自定义语音处理

```python
# 自定义VAD处理
async def custom_vad_process(audio_frame):
    # 实现自定义VAD逻辑
    energy = np.mean(np.abs(audio_frame))
    is_speech = energy > custom_threshold
    return {"is_speech": is_speech, "energy": energy}
```

### 扩展前端功能

```vue
<!-- 添加新的语音控件 -->
<template>
  <div class="custom-voice-control">
    <el-button @click="customVoiceAction">
      自定义功能
    </el-button>
  </div>
</template>
```

## 🔍 故障排除

### 常见问题

1. **TEN组件初始化失败**
   - 检查Python依赖是否完整安装
   - 确认TEN扩展路径正确
   - 查看后端日志获取详细错误

2. **WebSocket连接失败**
   - 确认后端服务正在运行
   - 检查防火墙设置
   - 验证端口8000是否可用

3. **麦克风权限问题**
   - 确保使用HTTPS或localhost
   - 检查浏览器麦克风权限设置
   - 尝试刷新页面重新授权

4. **语音检测不准确**
   - 调整VAD能量阈值
   - 检查音频采样率设置
   - 确认麦克风工作正常

### 调试技巧

```bash
# 查看后端日志
tail -f logs/backend.log

# 查看前端日志
tail -f logs/frontend.log

# 测试特定组件
python -c "from ten_vad_wrapper import TenVADWrapper; print('VAD OK')"
```

## 📈 性能优化

### 音频处理优化

- 使用AudioWorklet替代ScriptProcessor
- 实现音频缓冲区管理
- 优化WebSocket消息频率

### 内存管理

- 定期清理语音缓冲区
- 限制对话历史长度
- 实现连接池管理

## 🔮 未来规划

- [ ] 集成真实的Whisper ASR
- [ ] 添加TTS语音合成
- [ ] 支持多语言识别
- [ ] 实现语音情感分析
- [ ] 添加噪声抑制功能
- [ ] 支持多人对话场景

## 📞 支持

如有问题或建议，请：

1. 查看本文档的故障排除部分
2. 运行测试脚本诊断问题
3. 检查GitHub Issues
4. 提交新的Issue或PR

---

**Stream-Omni TEN框架集成** - 让AI对话更自然、更智能！ 🚀
