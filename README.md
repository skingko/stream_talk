# 🎤 Stream-Talk Voice Interaction System

基于实时语音交互的智能对话系统，集成了TEN框架、Spark-TTS、faster-whisper等先进技术，专为自然流畅的语音对话体验而设计。

## ✨ 主要特性

- 🗣️ **实时语音交互**: 支持连续对话，自然的语音交互体验，无限轮次多轮对话
- 🤖 **智能LLM集成**: 支持LM Studio + Qwen系列模型，提供强大的对话能力
- 🎵 **高质量TTS**: 使用Spark-TTS进行语音合成，支持流式音频生成和细粒度断句
- 🎯 **智能语音识别**: 集成faster-whisper large-v3-turbo模型，优化中文语音识别
- 🔧 **TEN框架**: 使用TEN VAD和Turn Detection进行智能语音处理
- 💻 **现代化界面**: 基于Vue3的响应式Web界面，支持语音和文本双模式
- 🍎 **macOS优化**: 完整支持Apple Silicon (M1/M2/M3) 和MPS加速
- 🔇 **回音抑制**: 智能说话人识别，防止AI语音被误识别为用户输入

## 🏗️ 系统架构

```
用户语音 → TEN VAD → faster-whisper → LM Studio (Qwen) → Spark-TTS → 语音输出
```


## 🚀 快速启动

### 1️⃣ 启动后端服务
```bash
python start_backend.py
```

### 2️⃣ 启动前端界面
```bash
cd frontend
npm run dev
```

### 3️⃣ 访问系统
- **前端界面**: http://localhost:5174
- **后端API**: http://localhost:8002

## 📋 环境要求

- **Python**: 3.9+
- **Node.js**: 16+
- **GPU**: CUDA/MPS支持 (推荐)
- **内存**: 16GB+
- **存储**: 足够空间存放AI模型

## 🎯 核心功能

### 🎤 语音交互
- 实时语音识别和合成，支持流式处理
- 智能回音抑制，防止AI语音干扰
- 自然的对话轮次管理和语音中断
- 支持无限轮次的连续对话

### 🤖 AI对话
- 基于LM Studio + Qwen系列大语言模型
- 智能的上下文理解和记忆
- 支持思维链推理（过滤</think>标签内容）

### 🎵 语音合成
- Spark-TTS高质量语音合成
- 细粒度断句分割（3个断句为一组）
- 流式音频生成，边生成边播放
- 优化的语音流畅度和自然度

### 🎯 语音识别
- faster-whisper large-v3-turbo模型
- 优化的中文语音识别
- int8量化，提升推理速度
- 支持beam_size=5的高质量转录

## 📁 项目结构

```
stream-talk/
├── simple_voice_server.py     # 主要后端服务
├── spark_tts_wrapper.py       # Spark-TTS包装器
├── third_party_paths.py       # 第三方库路径配置
├── start_backend.py           # 后端启动脚本
├── frontend/                  # Vue3前端应用
├── third-party/               # 第三方项目
│   ├── fish-speech/           # Fish Speech项目
│   ├── ten-framework/         # TEN框架
│   ├── ten-vad/              # VAD语音活动检测
│   └── ten-turn-detection/   # 轮次检测
├── models/                    # AI模型文件
├── assets/                    # 音频样本
└── docs/                      # 项目文档
```

详细结构说明: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## ⚙️ 系统要求

### 硬件要求
- **macOS**: Apple Silicon (M1/M2/M3) 推荐，支持MPS加速
- **内存**: 16GB+ 推荐（模型加载需要）
- **存储**: 20GB+ 可用空间（模型文件较大）

### 软件环境
- **Python**: 3.11+
- **Node.js**: 16+ (前端开发)
- **conda**: 用于环境管理

### LM Studio配置
- 地址: `http://localhost:1234`
- 推荐模型: Qwen2.5-7B-Instruct 或 Qwen2.5-14B-Instruct
- API兼容: OpenAI Chat Completions API

### 模型文件
- **Spark-TTS**: `models/spark-tts/` (MLX优化版本)
- **Whisper**: `models/whisper/` (faster-whisper large-v3-turbo)
- **参考音频**: `assets/音频样本*.wav`

## 🚀 安装指南

### 1. 环境准备
```bash
# 创建conda环境
conda create -n stream_omni python=3.11
conda activate stream_omni

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型准备
```bash
# Spark-TTS模型会在首次使用时自动下载
# faster-whisper模型会在首次使用时自动下载
# 确保有足够的磁盘空间（约10GB+）
```

### 3. LM Studio设置
1. 下载并安装 [LM Studio](https://lmstudio.ai/)
2. 下载Qwen2.5系列模型（推荐7B或14B版本）
3. 启动本地服务器，监听端口1234

### 4. 启动系统
```bash
# 启动后端服务
python start_backend.py

# 启动前端（新终端）
cd frontend
npm install
npm run dev
```

## 🎮 使用指南

1. **🎤 语音模式**: 点击麦克风按钮开始语音交互
   - 支持连续对话，无需重复点击
   - 智能回音抑制，避免AI语音干扰
   - 自动检测说话结束，流畅切换

2. **💬 文本模式**: 在输入框中输入文本进行对话
   - 支持Markdown格式显示
   - 实时流式响应

3. **🔄 模式切换**: 随时在语音和文本模式间切换
4. **⏹️ 中断功能**: 支持在AI说话时中断并重新开始

## 🛠️ 技术架构

### 核心组件
- **主服务**: `simple_voice_server.py` - WebSocket语音交互服务
- **TTS引擎**: `spark_tts_wrapper.py` - Spark-TTS包装器
- **路径管理**: `third_party_paths.py` - 第三方库路径配置
- **前端**: `frontend/src/` - Vue3组件和页面

### 第三方集成
- **TEN框架**: VAD和Turn Detection
- **faster-whisper**: 语音识别
- **Spark-TTS**: 语音合成
- **LM Studio**: LLM推理服务

## 📊 性能特性

- ⚡ **实时处理**: 流式音频传输和处理
- 🚀 **低延迟**: 优化的音频处理管道，<0.4s首帧延迟
- 🎯 **高质量**: Spark-TTS自然语音合成，支持中文优化
- 🧠 **智能检测**: TEN VAD和Turn Detection
- 🍎 **MPS加速**: 完整支持Apple Silicon GPU加速
- 🔇 **回音抑制**: 智能说话人识别，防止自激对话

## 📖 文档

- [项目结构说明](PROJECT_STRUCTURE.md)
- [语音交互系统指南](docs/Voice_Conversation_System_Guide.md)
- [TEN框架集成指南](docs/TEN_INTEGRATION_GUIDE.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件
