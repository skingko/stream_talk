<template>
  <div class="chat-input">
    <!-- 文本模式 -->
    <div v-if="mode === 'text'" class="text-input-container">
      <div class="input-wrapper">
        <!-- 文件上传按钮 -->
        <el-upload
          ref="uploadRef"
          :show-file-list="false"
          :before-upload="handleFileUpload"
          :accept="acceptedFileTypes"
          multiple
        >
          <el-button :icon="Paperclip" circle size="large" class="upload-btn" />
        </el-upload>

        <!-- 文本输入框 -->
        <el-input
          v-model="inputText"
          type="textarea"
          :rows="1"
          :autosize="{ minRows: 1, maxRows: 6 }"
          placeholder="输入消息..."
          @keydown.enter.exact.prevent="handleSend"
          @keydown.enter.shift.exact="handleNewLine"
          class="text-input"
          :disabled="disabled"
        />

        <!-- 发送按钮 -->
        <el-button
          :icon="Promotion"
          type="primary"
          circle
          size="large"
          @click="handleSend"
          :disabled="!inputText.trim() || disabled"
          class="send-btn"
        />
      </div>

      <!-- 已上传文件预览 -->
      <div v-if="uploadedFiles.length > 0" class="uploaded-files">
        <div v-for="file in uploadedFiles" :key="file.id" class="file-preview">
          <div class="file-info">
            <el-icon><Document /></el-icon>
            <span class="file-name">{{ file.name }}</span>
            <el-button
              :icon="Close"
              size="small"
              text
              @click="removeFile(file.id)"
              class="remove-btn"
            />
          </div>
          <img v-if="file.type === 'image'" :src="file.preview" class="image-preview" />
        </div>
      </div>
    </div>

    <!-- 语音模式 -->
    <div v-else class="voice-input-container">
      <!-- 实时语音交互界面 -->
      <div class="voice-interface">
        <!-- 中央可视化区域 -->
        <div class="voice-visualizer">
          <div class="visualizer-circle" :class="{ 
            'recording': isRecording, 
            'speaking': isSpeaking,
            'listening': isListening 
          }">
            <div class="inner-circle">
              <div class="wave-animation" v-if="isRecording || isSpeaking">
                <div class="wave" v-for="i in 3" :key="i" :style="{ animationDelay: i * 0.1 + 's' }"></div>
              </div>
            </div>
          </div>
        </div>

        <!-- 状态文本 -->
        <div class="voice-status">
          <span v-if="isListening">正在监听...</span>
          <span v-else-if="isRecording">正在录音...</span>
          <span v-else-if="isSpeaking">正在播放...</span>
          <span v-else-if="isProcessing">正在处理...</span>
          <span v-else>点击开始语音对话</span>
        </div>

        <!-- 实时转录文本 -->
        <div v-if="realtimeTranscript" class="realtime-transcript">
          {{ realtimeTranscript }}
        </div>

        <!-- 控制按钮 -->
        <div class="voice-controls">
          <!-- 录音/停止按钮 -->
          <el-button
            :icon="isRecording ? VideoPause : Microphone"
            :type="isRecording ? 'danger' : 'primary'"
            circle
            size="large"
            @click="toggleRecording"
            :disabled="disabled || isProcessing"
            class="record-btn"
          />

          <!-- 静音/取消静音按钮 -->
          <el-button
            :icon="isMuted ? MuteNotification : Microphone"
            circle
            size="large"
            @click="handleToggleMute"
            class="mute-btn"
          />

          <!-- 停止播放/打断按钮 -->
          <el-button
            v-if="isSpeaking"
            :icon="Close"
            type="danger"
            circle
            size="large"
            @click="handleStopSpeaking"
            class="stop-btn"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Paperclip, Promotion, Document, Close, Microphone,
  VideoPause, MuteNotification
} from '@element-plus/icons-vue'
import { useVoice } from '../composables/useVoice'

// Props
const props = defineProps({
  mode: {
    type: String,
    default: 'text' // 'text' | 'voice'
  },
  disabled: {
    type: Boolean,
    default: false
  }
})

// Emits
const emit = defineEmits([
  'send-message',
  'upload-file', 
  'voice-input',
  'stop-recording',
  'stop-speaking'
])

// 响应式数据
const inputText = ref('')
const uploadedFiles = ref([])
const uploadRef = ref(null)

// 使用语音功能
const { voiceState, startRecording, stopRecording, stopSpeaking, toggleMute } = useVoice()

// 语音状态的计算属性
const isRecording = computed(() => voiceState.isRecording)
const isListening = computed(() => voiceState.isListening)
const isSpeaking = computed(() => voiceState.isSpeaking)
const isProcessing = computed(() => voiceState.isProcessing)
const isMuted = computed(() => voiceState.isMuted)
const realtimeTranscript = computed(() => voiceState.transcript)

// 文件类型限制
const acceptedFileTypes = '.jpg,.jpeg,.png,.gif,.mp4,.mov,.avi,.mp3,.wav,.pdf,.doc,.docx,.txt'

// 计算属性
const canSend = computed(() => {
  return inputText.value.trim() || uploadedFiles.value.length > 0
})

// 方法
const handleSend = () => {
  if (!canSend.value || props.disabled) return

  const message = {
    type: uploadedFiles.value.length > 0 ? 'multimodal' : 'text',
    content: inputText.value.trim(),
    files: uploadedFiles.value
  }

  emit('send-message', message)
  
  // 清空输入
  inputText.value = ''
  uploadedFiles.value = []
}

const handleNewLine = () => {
  inputText.value += '\n'
}

const handleFileUpload = (file) => {
  const fileId = Date.now() + Math.random()
  const fileType = getFileType(file.type)
  
  const fileInfo = {
    id: fileId,
    name: file.name,
    type: fileType,
    file: file,
    size: file.size
  }

  // 如果是图片，生成预览
  if (fileType === 'image') {
    const reader = new FileReader()
    reader.onload = (e) => {
      fileInfo.preview = e.target.result
    }
    reader.readAsDataURL(file)
  }

  uploadedFiles.value.push(fileInfo)
  emit('upload-file', fileInfo)
  
  return false // 阻止自动上传
}

const removeFile = (fileId) => {
  uploadedFiles.value = uploadedFiles.value.filter(f => f.id !== fileId)
}

const getFileType = (mimeType) => {
  if (mimeType.startsWith('image/')) return 'image'
  if (mimeType.startsWith('video/')) return 'video'
  if (mimeType.startsWith('audio/')) return 'audio'
  return 'file'
}

// 语音相关方法
const toggleRecording = () => {
  if (isRecording.value) {
    stopRecording()
  } else {
    startRecording()
  }
}

const handleStopSpeaking = () => {
  stopSpeaking()
  emit('stop-speaking')
}

const handleToggleMute = () => {
  toggleMute()
}

// 监听模式变化
watch(() => props.mode, (newMode) => {
  if (newMode === 'voice') {
    // 切换到语音模式时，清空文本输入
    inputText.value = ''
    uploadedFiles.value = []
  } else {
    // 切换到文本模式时，停止所有语音活动
    if (isRecording.value) {
      stopRecording()
    }
    if (isSpeaking.value) {
      stopSpeaking()
    }
  }
})
</script>

<style lang="scss" scoped>
.chat-input {
  width: 100%;
}

.text-input-container {
  .input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 12px;
    padding: 16px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .upload-btn {
    flex-shrink: 0;
    color: #6b7280;
    
    &:hover {
      color: #3b82f6;
    }
  }

  .text-input {
    flex: 1;
    
    :deep(.el-textarea__inner) {
      border: none;
      box-shadow: none;
      resize: none;
      padding: 8px 0;
      font-size: 16px;
      line-height: 1.5;
      
      &:focus {
        border: none;
        box-shadow: none;
      }
    }
  }

  .send-btn {
    flex-shrink: 0;
  }
}

.uploaded-files {
  margin-top: 12px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.file-preview {
  background: #f3f4f6;
  border-radius: 8px;
  padding: 8px;
  max-width: 200px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.file-name {
  flex: 1;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-preview {
  width: 100%;
  height: 100px;
  object-fit: cover;
  border-radius: 4px;
}

.voice-input-container {
  display: flex;
  justify-content: center;
  padding: 32px;
}

.voice-interface {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  max-width: 400px;
  width: 100%;
}

.voice-visualizer {
  position: relative;
  width: 200px;
  height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.visualizer-circle {
  width: 160px;
  height: 160px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  position: relative;
  
  &.recording {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    transform: scale(1.1);
  }
  
  &.speaking {
    background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
    animation: pulse 2s infinite;
  }
  
  &.listening {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    animation: breathe 3s infinite;
  }
}

.inner-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.wave-animation {
  position: absolute;
  width: 100%;
  height: 100%;
}

.wave {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.6);
  animation: wave 2s infinite;
}

.voice-status {
  font-size: 18px;
  font-weight: 500;
  color: #374151;
  text-align: center;
}

.realtime-transcript {
  background: #f3f4f6;
  padding: 16px;
  border-radius: 12px;
  font-size: 16px;
  color: #374151;
  text-align: center;
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
}

.voice-controls {
  display: flex;
  gap: 16px;
  align-items: center;
}

.record-btn {
  width: 64px;
  height: 64px;
  font-size: 24px;
}

.mute-btn, .stop-btn {
  width: 48px;
  height: 48px;
  font-size: 20px;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

@keyframes breathe {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.02);
    opacity: 0.8;
  }
}

@keyframes wave {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  100% {
    transform: scale(1.4);
    opacity: 0;
  }
}

@media (max-width: 768px) {
  .voice-visualizer {
    width: 150px;
    height: 150px;
  }
  
  .visualizer-circle {
    width: 120px;
    height: 120px;
  }
  
  .inner-circle {
    width: 90px;
    height: 90px;
  }
  
  .record-btn {
    width: 56px;
    height: 56px;
    font-size: 20px;
  }
  
  .mute-btn, .stop-btn {
    width: 44px;
    height: 44px;
    font-size: 18px;
  }
}
</style>
