# Stream-Talk 项目结构

## 📁 清理后的目录结构

```
stream-talk/
├── 🚀 核心服务
│   ├── simple_voice_server.py          # 主要后端服务器
│   ├── spark_tts_wrapper.py            # Spark-TTS包装器
│   ├── start_backend.py                # 后端启动脚本
│   └── start_all.py                    # 启动所有服务
│
├── 🎨 前端界面
│   └── frontend/                       # Vue3前端应用
│       ├── src/                        # 源代码
│       │   ├── components/             # Vue组件
│       │   └── composables/            # 组合式函数
│       ├── package.json                # 依赖配置
│       └── start.sh                    # 前端启动脚本
│
├── 🤖 AI模型
│   └── models/                         # 模型文件
│       ├── fish-speech/                # Fish Speech TTS模型
│       ├── spark-tts/                  # Spark-TTS模型
│       ├── stream-omni-8b/             # Stream-Omni 8B模型
│       └── whisper/                    # Whisper语音识别模型
│
├── 🔧 TEN框架组件
│   ├── third-party/                    # 第三方项目
│   │   ├── ten-framework/              # TEN框架核心
│   │   ├── ten-vad/                    # VAD语音活动检测
│   │   ├── ten-turn-detection/         # Turn Detection轮次检测
│   │   └── fish-speech/                # Fish Speech项目
│   └── extensions/                     # TEN扩展组件
│       ├── conversation_manager/       # 对话管理
│       ├── fish_speech_tts/           # Fish Speech TTS扩展
│       ├── lm_studio_llm/             # LM Studio LLM扩展
│       ├── ten_turn_detection/        # Turn Detection扩展
│       └── ten_vad/                   # VAD扩展
│
├── 🎵 音频资源
│   └── assets/                         # 静态资源
│       ├── model.png                   # 模型架构图
│       ├── stream-omni.png             # 项目logo
│       ├── 音频样本1.-温婉女声wav.wav    # 参考音频样本
│       └── 音频样本2-可爱女声.wav       # 备用音频样本
│
├── 📚 文档
│   └── docs/                           # 项目文档
│       ├── Voice_Conversation_System_Guide.md
│       ├── TEN_INTEGRATION_GUIDE.md
│       ├── Project_Final_Summary.md
│       ├── SPARK_TTS_INTEGRATION.md
│       └── Stream-Omni_Implementation_Summary.md
│
├── ⚙️ 配置文件
│   ├── requirements.txt                # Python依赖
│   ├── pyproject.toml                 # 项目配置
│   ├── activate_env.sh                 # 环境激活脚本
│   └── stream_omni/                   # 核心模块
│
└── 📄 项目信息
    ├── README.md                      # 项目说明
    ├── LICENSE                        # 许可证
    └── PROJECT_STRUCTURE.md           # 项目结构说明
```

## 🎯 核心功能模块

### 1. 语音交互服务 (`simple_voice_server.py`)
- **WebSocket实时通信**
- **faster-whisper语音识别** (large-v3-turbo, int8量化)
- **LM Studio + Qwen LLM对话**
- **Spark-TTS语音合成** (MLX优化)
- **VAD语音活动检测**
- **Turn Detection轮次判断**
- **智能回音抑制**

### 2. TTS引擎 (`spark_tts_wrapper.py`)
- **Spark-TTS MLX优化版本**
- **流式音频生成**
- **细粒度断句分割** (3个断句为一组)
- **持久化模型实例**
- **MPS GPU加速支持**

### 3. 前端界面 (`frontend/`)
- **Vue3 + Element Plus**
- **实时语音交互界面**
- **文本/语音模式切换**
- **流式音频播放**
- **对话历史管理**
- **自动重连机制**

### 4. AI模型集成
- **Qwen2.5系列**: 大语言模型 (通过LM Studio)
- **Spark-TTS**: 高质量TTS语音合成 (MLX优化)
- **faster-whisper large-v3-turbo**: 语音识别 (int8量化)
- **TEN VAD**: 语音活动检测
- **TEN Turn Detection**: 对话轮次检测

### 5. 第三方库管理 (`third_party_paths.py`)
- **统一路径配置**
- **自动路径检测**
- **导入路径管理**
- **库可用性验证**

## 🚀 快速启动

### 启动后端服务
```bash
python start_backend.py
```

### 启动前端界面
```bash
cd frontend
npm run dev
# 或者
./start.sh
```

## 📋 系统要求

### 硬件要求
- **macOS**: Apple Silicon (M1/M2/M3) 推荐，支持MPS加速
- **内存**: 16GB+ 推荐 (模型加载需要)
- **存储**: 20GB+ 可用空间 (模型文件较大)

### 软件环境
- **Python**: 3.11+
- **Node.js**: 16+ (前端开发)
- **conda**: 用于环境管理

### 模型要求
- **LM Studio**: 本地LLM推理服务
- **Qwen2.5**: 7B或14B版本推荐
- **Spark-TTS**: MLX优化版本，自动下载
- **faster-whisper**: large-v3-turbo，自动下载

## 🔗 访问地址

- **前端界面**: http://localhost:5173
- **后端API**: http://localhost:8002
- **WebSocket**: ws://localhost:8002/ws/voice
- **LM Studio**: http://localhost:1234

## 📝 注意事项

1. **macOS优化**: 项目专为Apple Silicon优化，支持MPS GPU加速
2. **模型自动下载**: Spark-TTS和faster-whisper模型首次使用时自动下载
3. **参考音频**: `assets/` 目录中的音频文件用于TTS音色参考
4. **回音抑制**: 系统自动处理AI语音回音，避免自激对话
5. **内存管理**: 使用持久化模型实例，减少重复加载

## 🛠️ 开发说明

- **核心服务器**: `simple_voice_server.py` 是主要的后端服务
- **TTS包装器**: `spark_tts_wrapper.py` 处理Spark-TTS语音合成
- **路径管理**: `third_party_paths.py` 统一管理第三方库路径
- **前端组件**: 位于 `frontend/src/` 目录
- **扩展开发**: 可在 `extensions/` 目录添加新功能

## 📊 性能优化

### 🚀 核心优化
- **MPS GPU加速**: 完整支持Apple Silicon GPU加速
- **流式处理**: 音频数据采用流式传输，边生成边播放
- **持久化实例**: TTS模型使用持久化实例，避免重复加载
- **异步处理**: 所有IO操作均为异步，提升并发性能

### 🎵 TTS优化
- **MLX框架**: 使用MLX优化的Spark-TTS，专为Apple Silicon设计
- **细粒度分割**: 3个断句为一组，优化语音流畅度
- **实时率**: <0.4s首帧延迟，实时率0.44x-0.97x
- **内存优化**: 智能缓存和内存管理

### 🎯 ASR优化
- **int8量化**: faster-whisper使用int8量化，提升推理速度
- **beam搜索**: beam_size=5，平衡速度和质量
- **中文优化**: 专门优化中文语音识别准确率

### 🔇 智能优化
- **回音抑制**: 时间窗口+说话人识别，防止AI语音干扰
- **自动重连**: WebSocket自动重连，保证连接稳定性
- **状态管理**: 智能对话状态管理，支持无限轮次对话
