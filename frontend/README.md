# Stream-Omni 前端界面

基于 Vue3 + Element Plus 的 ChatGPT 风格多模态AI对话界面。

## 功能特性

### 🎯 核心功能
- **智能对话**: 支持文本对话，流式响应
- **多模态交互**: 支持图片、视频、音频文件上传和理解
- **实时语音**: 语音输入输出，支持打断LLM回复
- **模式切换**: 文本模式和语音模式无缝切换

### 🎨 界面特性
- **ChatGPT风格**: 现代化的对话界面设计
- **响应式布局**: 适配桌面和移动设备
- **实时可视化**: 语音模式下的动态可视化效果
- **流畅动画**: 丰富的交互动画和过渡效果

### 🎤 语音功能
- **实时录音**: 高质量音频录制
- **语音识别**: 实时语音转文本
- **语音合成**: 文本转语音播放
- **打断功能**: 支持打断AI语音输出
- **音量监测**: 实时音量可视化

## 技术栈

- **框架**: Vue 3 (Composition API)
- **UI库**: Element Plus
- **构建工具**: Vite
- **样式**: SCSS
- **HTTP客户端**: Axios
- **语音处理**: Web Audio API, MediaRecorder API

## 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 启动开发服务器

```bash
# 方式1: 使用npm命令
npm run dev

# 方式2: 使用启动脚本
./start.sh
```

### 3. 访问应用

打开浏览器访问: http://localhost:5173

## 项目结构

```
frontend/
├── src/
│   ├── components/          # Vue组件
│   │   ├── ChatMessages.vue # 消息显示组件
│   │   └── ChatInput.vue    # 输入组件
│   ├── composables/         # 组合式函数
│   │   ├── useChat.js       # 聊天功能
│   │   └── useVoice.js      # 语音功能
│   ├── styles/              # 样式文件
│   │   └── global.scss      # 全局样式
│   ├── App.vue              # 主应用组件
│   └── main.js              # 应用入口
├── index.html               # HTML模板
├── package.json             # 项目配置
├── vite.config.js           # Vite配置
└── start.sh                 # 启动脚本
```

## 使用说明

### 文本模式
1. 在侧边栏选择"💬"切换到文本模式
2. 在输入框中输入文本消息
3. 点击📎按钮上传图片、视频等文件
4. 按Enter发送消息，Shift+Enter换行

### 语音模式
1. 在侧边栏选择"🎤"切换到语音模式
2. 点击中央录音按钮开始语音输入
3. 说话时会看到实时的音频可视化效果
4. 再次点击停止录音，系统会自动识别并回复
5. 在AI回复时可以点击❌按钮打断输出

### 多模态交互
- 上传图片: 支持JPG、PNG、GIF格式
- 上传视频: 支持MP4、MOV、AVI格式
- 上传音频: 支持MP3、WAV格式
- 上传文档: 支持PDF、DOC、TXT格式

## API接口

前端会连接到后端API服务 (默认: http://localhost:8000):

- `POST /api/chat` - 文本对话
- `POST /api/upload` - 文件上传
- `POST /api/speech-to-text` - 语音识别
- `POST /api/text-to-speech` - 语音合成
- `POST /api/image-text` - 图像理解
- `POST /api/multimodal` - 多模态交互
- `GET /health` - 健康检查

## 浏览器兼容性

- Chrome 88+
- Firefox 85+
- Safari 14+
- Edge 88+

**注意**: 语音功能需要HTTPS环境或localhost才能正常工作。

## 开发说明

### 环境要求
- Node.js 16+
- npm 8+

### 开发命令
```bash
npm run dev      # 启动开发服务器
npm run build    # 构建生产版本
npm run preview  # 预览生产版本
```

### 自定义配置

可以在 `src/composables/useChat.js` 中修改API基础URL:

```javascript
const api = axios.create({
  baseURL: 'http://your-api-server:8000',
  // ...
})
```

## 故障排除

### 语音功能不工作
1. 确保浏览器支持Web Audio API
2. 检查麦克风权限是否已授予
3. 确保在HTTPS环境或localhost下运行

### 文件上传失败
1. 检查文件大小是否超过限制
2. 确认文件格式是否支持
3. 检查后端API服务是否正常运行

### 连接后端失败
1. 确认后端服务已启动 (http://localhost:8000)
2. 检查CORS配置是否正确
3. 查看浏览器控制台的错误信息

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

MIT License
