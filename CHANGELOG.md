# 📝 Stream-Talk 更新日志

## [2.0.0] - 2025-01-05

### 🎉 重大更新
- **项目重命名**: Stream-Omni → Stream-Talk
- **TTS引擎切换**: Fish Speech → Spark-TTS (MLX优化)
- **ASR引擎优化**: Whisper → faster-whisper (large-v3-turbo)
- **LLM集成**: 主要支持LM Studio + Qwen系列模型

### ✨ 新功能
- **🍎 macOS优化**: 完整支持Apple Silicon (M1/M2/M3) 和MPS加速
- **🔇 回音抑制**: 智能说话人识别，防止AI语音被误识别为用户输入
- **🎵 细粒度断句**: 3个断句为一组的优化语音输出模式
- **🔄 自动重连**: WebSocket连接自动重连机制
- **📦 路径管理**: 统一的第三方库路径配置系统

### 🚀 性能优化
- **实时率提升**: TTS首帧延迟 <0.4s，实时率0.44x-0.97x
- **内存优化**: 持久化模型实例，减少重复加载
- **GPU加速**: 完整的MPS GPU加速支持
- **流式处理**: 边生成边播放的音频流处理

### 🔧 技术改进
- **faster-whisper**: 使用large-v3-turbo模型，int8量化
- **Spark-TTS**: MLX优化版本，专为Apple Silicon设计
- **细粒度分割**: 支持逗号、分号等标点的细粒度断句
- **智能缓存**: MLX kernel缓存，提升推理性能

### 📁 项目结构优化
- **第三方库管理**: 统一移动到`third-party/`目录
- **路径配置**: 新增`third_party_paths.py`统一管理路径
- **文档更新**: 完整更新README和项目结构文档
- **代码清理**: 移除无用的测试文件和临时代码

### 🐛 问题修复
- **导入路径**: 修复第三方库导入路径问题
- **连接稳定性**: 改善WebSocket连接稳定性
- **音频播放**: 修复音频播放完成检测问题
- **状态管理**: 优化对话状态管理逻辑

### 📚 文档更新
- **README.md**: 完整重写，反映当前技术栈
- **PROJECT_STRUCTURE.md**: 更新项目结构和技术说明
- **系统要求**: 明确macOS和Apple Silicon支持
- **安装指南**: 详细的安装和配置步骤

### 🔄 迁移指南
如果您从旧版本升级：

1. **环境重建**:
   ```bash
   conda create -n stream_omni python=3.11
   conda activate stream_omni
   pip install -r requirements.txt
   ```

2. **模型更新**:
   - Spark-TTS模型会自动下载
   - faster-whisper模型会自动下载
   - 移除旧的Fish Speech模型文件

3. **配置更新**:
   - 确保LM Studio运行在localhost:1234
   - 下载Qwen2.5系列模型
   - 检查参考音频文件位置

### 🎯 下一步计划
- [ ] 支持更多TTS引擎选择
- [ ] 添加语音情感识别
- [ ] 支持多语言切换
- [ ] 优化移动端支持
- [ ] 添加语音克隆功能

---

## [1.0.0] - 2024-12-XX

### 🎉 初始版本
- 基础语音交互系统
- Fish Speech TTS集成
- Whisper ASR集成
- TEN框架集成
- Vue3前端界面
