# Spark-TTS 集成文档

## 概述

Stream-Omni 项目已成功集成 Spark-TTS 0.5B (MLX/6-bit) 推理引擎，提供高质量的流式语音合成功能。

## 功能特性

- ✅ **流式音频生成**: 基于句级切分+并行推理实现实时音频输出
- ✅ **温柔女声**: 使用 af_heart 声音配置，提供温暖自然的女声输出
- ✅ **情感标记处理**: 自动移除不支持的情感标记，确保兼容性
- ✅ **速度映射**: 自动将任意速度值映射到 Spark-TTS 支持的速度值
- ✅ **本地模型**: 使用本地下载的模型，无需网络连接

## 模型信息

- **模型路径**: `models/spark-tts/Spark-TTS-0.5B-6bit`
- **采样率**: 24000 Hz
- **支持速度**: 0.0 (very_low), 0.5 (low), 1.0 (moderate), 1.5 (high), 2.0 (very_high)
- **声音配置**: af_heart (温柔女声)

## 使用方法

### 启动后端服务

```bash
# 使用 conda 环境
conda activate stream_omni

# 启动后端服务
python start_backend.py

# 或直接启动
python simple_voice_server.py
```

### API 接口

后端服务提供以下接口：

- **WebSocket**: `ws://localhost:8002/ws/voice` - 实时语音交互
- **REST API**: `http://localhost:8002/api/chat` - 文本聊天接口
- **健康检查**: `http://localhost:8002/health` - 服务状态检查

### 前端使用

前端已集成 WebSocket 连接，支持：

- 实时语音识别 (Whisper)
- 流式音频播放 (Spark-TTS)
- 语音中断和恢复
- 对话状态管理

## 配置说明

### 温柔女声配置

```python
voice_settings = {
    "gender": "female",
    "pitch": -2.0,      # 降低音调，更温柔
    "speed": 1.0,       # 正常语速
    "temperature": 0.7, # 适中的随机性
    "voice": "af_heart" # 温柔女声
}
```

### 参考音频

- **文件路径**: `assets/音频样本1.-温婉女声wav.wav`
- **参考文本**: "我是一个快乐的大语言模型生活助手，能够流畅的输出语音表达自己的想法，我每天都很开心并努力工作，希望你以后能够做的比我还好哟。"

## 性能优化

### M3 Max 优化配置

- **设备**: 自动检测 (MPS/CPU)
- **精度**: float16 (GPU) / float32 (CPU)
- **并行处理**: 句级切分 + 多线程生成
- **内存优化**: 及时清理临时文件

### 实时性能

- **延迟**: < 2秒 (短句)
- **吞吐量**: ~5 tokens/sec
- **内存使用**: ~2GB (模型加载)

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否存在: `models/spark-tts/Spark-TTS-0.5B-6bit`
   - 重新下载模型: `huggingface-cli download mlx-community/Spark-TTS-0.5B-6bit`

2. **音频生成卡死**
   - 检查是否有情感标记: 自动移除 `(emotion)` 标记
   - 检查速度值: 自动映射到支持的速度值

3. **WebSocket 连接失败**
   - 检查端口占用: `lsof -i:8002`
   - 重启服务: `python simple_voice_server.py`

### 日志调试

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 技术架构

```
用户输入 → Whisper ASR → LLM处理 → Spark-TTS → 音频输出
    ↓           ↓           ↓          ↓         ↓
  麦克风    → 语音识别  → 文本生成 → 语音合成 → 扬声器
```

### 组件说明

- **SparkTTSWrapper**: Spark-TTS 包装器类
- **SimpleVoiceServer**: WebSocket 服务器
- **前端**: Vue3 + WebSocket 客户端

## 更新日志

### v1.0.0 (2025-07-05)

- ✅ 集成 Spark-TTS 0.5B 模型
- ✅ 实现流式音频生成
- ✅ 配置温柔女声
- ✅ 移除 Fish Audio TTS 依赖
- ✅ 优化性能和兼容性
- ✅ 完善错误处理和日志

## 许可证

本项目遵循原 Stream-Omni 项目的许可证。
