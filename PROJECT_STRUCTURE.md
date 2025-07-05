# Stream-Omni 项目结构

## 📁 清理后的目录结构

```
streem-omni/
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
- **Whisper语音识别**
- **LM Studio LLM对话**
- **Fish Audio TTS语音合成**
- **VAD语音活动检测**
- **Turn Detection轮次判断**

### 2. 前端界面 (`frontend/`)
- **Vue3 + Element Plus**
- **实时语音交互界面**
- **文本/语音模式切换**
- **流式音频播放**
- **对话历史管理**

### 3. AI模型集成
- **Stream-Omni 8B**: 多模态大语言模型
- **Fish Speech**: 高质量TTS语音合成
- **Whisper Large-v3**: 语音识别
- **TEN VAD**: 语音活动检测
- **TEN Turn Detection**: 对话轮次检测

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

## 📋 依赖要求

- **Python 3.9+**
- **Node.js 16+**
- **CUDA支持的GPU** (推荐)
- **16GB+ RAM**
- **足够的存储空间** (模型文件较大)

## 🔗 访问地址

- **前端界面**: http://localhost:5174
- **后端API**: http://localhost:8002
- **WebSocket**: ws://localhost:8002/ws/voice

## 📝 注意事项

1. **模型文件**: 确保所有AI模型已正确下载到 `models/` 目录
2. **参考音频**: `assets/` 目录中的音频文件用于TTS音色克隆
3. **GPU加速**: 建议使用GPU以获得最佳性能
4. **内存需求**: 多个大模型同时运行需要充足内存

## 🛠️ 开发说明

- **核心服务器**: `simple_voice_server.py` 是主要的后端服务
- **TTS包装器**: `fish_audio_tts_wrapper.py` 处理语音合成
- **前端组件**: 位于 `frontend/src/` 目录
- **扩展开发**: 可在 `extensions/` 目录添加新功能

## 📊 性能优化

- **流式处理**: 音频数据采用流式传输
- **缓存机制**: TTS模型使用缓存提高效率
- **异步处理**: 所有IO操作均为异步
- **内存管理**: 自动清理连接和缓存数据
