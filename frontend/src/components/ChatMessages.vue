<template>
  <div class="chat-messages" ref="messagesContainer">
    <div class="messages-wrapper">
      <!-- 欢迎消息 -->
      <div v-if="messages.length === 0" class="welcome-message">
        <div class="welcome-content">
          <div class="welcome-icon">
            <div class="logo-emoji">🎙️</div>
          </div>
          <h2>欢迎使用 Stream-Omni</h2>
          <p>我是一个多模态AI助手，可以处理文本、图像、音频和视频。</p>
          <div class="feature-cards">
            <div class="feature-card">
              <el-icon><ChatDotRound /></el-icon>
              <span>智能对话</span>
            </div>
            <div class="feature-card">
              <el-icon><Picture /></el-icon>
              <span>图像理解</span>
            </div>
            <div class="feature-card">
              <el-icon><Microphone /></el-icon>
              <span>语音交互</span>
            </div>
            <div class="feature-card">
              <el-icon><VideoCamera /></el-icon>
              <span>视频分析</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 消息列表 -->
      <div v-for="message in messages" :key="message.id" class="message-item">
        <div class="message-wrapper" :class="{ 'user-message': message.role === 'user' }">
          <!-- 头像 -->
          <div class="avatar">
            <div v-if="message.role === 'user'" class="user-avatar">
              <el-icon><User /></el-icon>
            </div>
            <div v-else class="assistant-avatar">
              <div class="assistant-emoji">🤖</div>
            </div>
          </div>

          <!-- 消息内容 -->
          <div class="message-content">
            <div class="message-header">
              <span class="sender-name">
                {{ message.role === 'user' ? '你' : 'Stream-Omni' }}
              </span>
              <span class="message-time">
                {{ formatTime(message.timestamp) }}
              </span>
            </div>

            <div class="message-body">
              <!-- 文本消息 -->
              <div v-if="message.type === 'text'" class="text-message">
                <div v-html="formatMessage(message.content)"></div>
              </div>

              <!-- 图片消息 -->
              <div v-else-if="message.type === 'image'" class="image-message">
                <el-image 
                  :src="message.content" 
                  :preview-src-list="[message.content]"
                  fit="cover"
                  class="message-image"
                />
              </div>

              <!-- 语音消息显示为文本转录 -->
              <div v-else-if="message.type === 'voice'" class="voice-message">
                <div class="voice-indicator">
                  <el-icon><Microphone /></el-icon>
                  <span>语音消息</span>
                </div>
                <div v-if="message.transcript" class="voice-transcript">
                  {{ message.transcript }}
                </div>
              </div>

              <!-- 视频消息 -->
              <div v-else-if="message.type === 'video'" class="video-message">
                <video :src="message.content" controls class="message-video" />
              </div>

              <!-- 文件消息 -->
              <div v-else-if="message.type === 'file'" class="file-message">
                <div class="file-info">
                  <el-icon><Document /></el-icon>
                  <span>{{ message.fileName }}</span>
                  <el-button size="small" type="primary" @click="downloadFile(message.content)">
                    下载
                  </el-button>
                </div>
              </div>

              <!-- 多模态消息 -->
              <div v-else-if="message.type === 'multimodal'" class="multimodal-message">
                <div v-if="message.content.text" class="text-part">
                  <div v-html="formatMessage(message.content.text)"></div>
                </div>
                <div v-if="message.content.image" class="image-part">
                  <el-image 
                    :src="message.content.image" 
                    :preview-src-list="[message.content.image]"
                    fit="cover"
                    class="message-image"
                  />
                </div>

              </div>
            </div>

            <!-- 消息操作 -->
            <div class="message-actions" v-if="message.role === 'assistant'">
              <el-button size="small" text @click="copyMessage(message.content)">
                <el-icon><CopyDocument /></el-icon>
                复制
              </el-button>
              <el-button size="small" text @click="regenerateMessage(message)">
                <el-icon><Refresh /></el-icon>
                重新生成
              </el-button>

            </div>
          </div>
        </div>
      </div>

      <!-- 加载状态 -->
      <div v-if="loading" class="loading-message">
        <div class="message-wrapper">
          <div class="avatar">
            <div class="assistant-avatar">
              <div class="assistant-emoji">🤖</div>
            </div>
          </div>
          <div class="message-content">
            <div class="message-header">
              <span class="sender-name">Stream-Omni</span>
            </div>
            <div class="message-body">
              <div class="typing-indicator">
                <div class="typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
                <span class="typing-text">正在思考中...</span>
                <el-button 
                  size="small" 
                  type="danger" 
                  @click="$emit('stop-generation')"
                  class="stop-btn"
                >
                  <el-icon><Close /></el-icon>
                  停止生成
                </el-button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { 
  ChatDotRound, Picture, Microphone, VideoCamera, User, 
  Document, CopyDocument, Refresh, Close 
} from '@element-plus/icons-vue'

import { marked } from 'marked'

// Props
const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  },
  loading: {
    type: Boolean,
    default: false
  },
  currentMode: {
    type: String,
    default: 'text'
  }
})

// Emits
const emit = defineEmits(['stop-generation'])

// Refs
const messagesContainer = ref(null)

// 方法
const formatTime = (timestamp) => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}

const formatMessage = (content) => {
  if (typeof content !== 'string') return content
  return marked(content, { breaks: true })
}

const copyMessage = async (content) => {
  try {
    await navigator.clipboard.writeText(content)
    ElMessage.success('已复制到剪贴板')
  } catch (error) {
    ElMessage.error('复制失败')
  }
}

const regenerateMessage = (message) => {
  console.log('重新生成消息:', message)
  // TODO: 实现重新生成逻辑
}



const downloadFile = (url) => {
  const link = document.createElement('a')
  link.href = url
  link.download = ''
  link.click()
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 监听消息变化，自动滚动到底部
watch(() => props.messages, scrollToBottom, { deep: true })
watch(() => props.loading, scrollToBottom)
</script>

<style lang="scss" scoped>
.chat-messages {
  height: 100%;
  overflow-y: auto;
  padding: 16px;
}

.messages-wrapper {
  max-width: 800px;
  margin: 0 auto;
}

.welcome-message {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 60vh;
  text-align: center;
}

.welcome-content {
  .welcome-icon {
    margin-bottom: 24px;
    display: flex;
    justify-content: center;

    .logo-emoji {
      font-size: 48px;
      line-height: 1;
    }

    img {
      width: 64px;
      height: 64px;
    }
  }
  
  h2 {
    font-size: 28px;
    font-weight: 600;
    margin-bottom: 12px;
    color: #1f2937;
  }
  
  p {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 32px;
  }
}

.feature-cards {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  max-width: 400px;
  margin: 0 auto;
}

.feature-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
  
  &:hover {
    transform: translateY(-2px);
  }
  
  .el-icon {
    font-size: 24px;
    color: #3b82f6;
  }
  
  span {
    font-size: 14px;
    font-weight: 500;
    color: #374151;
  }
}

.message-item {
  margin-bottom: 24px;
}

.message-wrapper {
  display: flex;
  gap: 12px;
  
  &.user-message {
    flex-direction: row-reverse;
    
    .message-content {
      background: #3b82f6;
      color: white;
      border-radius: 18px 18px 4px 18px;
    }
  }
}

.avatar {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
}

.user-avatar {
  width: 100%;
  height: 100%;
  background: #6b7280;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.assistant-avatar {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

  .assistant-emoji {
    font-size: 20px;
    line-height: 1;
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
}

.message-content {
  flex: 1;
  background: white;
  border-radius: 18px 18px 18px 4px;
  padding: 12px 16px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
}

.message-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.sender-name {
  font-weight: 600;
  font-size: 14px;
}

.message-time {
  font-size: 12px;
  color: #6b7280;
}

.message-body {
  margin-bottom: 8px;
}

.text-message {
  line-height: 1.6;
  
  :deep(p) {
    margin-bottom: 8px;
    
    &:last-child {
      margin-bottom: 0;
    }
  }
  
  :deep(code) {
    background: #f3f4f6;
    padding: 2px 4px;
    border-radius: 4px;
    font-family: 'Monaco', 'Consolas', monospace;
  }
  
  :deep(pre) {
    background: #f3f4f6;
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 8px 0;
  }
}

.voice-message {
  .voice-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #3b82f6;
    font-size: 14px;
    margin-bottom: 8px;
  }

  .voice-transcript {
    font-style: italic;
    color: #6b7280;
  }
}

.message-image {
  max-width: 300px;
  border-radius: 8px;
}

.message-video {
  max-width: 400px;
  border-radius: 8px;
}

.file-info {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: #f3f4f6;
  border-radius: 8px;
}

.message-actions {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.typing-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
}

.typing-dots {
  display: flex;
  gap: 4px;
  
  span {
    width: 6px;
    height: 6px;
    background: #3b82f6;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
    
    &:nth-child(1) { animation-delay: -0.32s; }
    &:nth-child(2) { animation-delay: -0.16s; }
  }
}

.typing-text {
  color: #6b7280;
  font-size: 14px;
}

.stop-btn {
  margin-left: auto;
}

@keyframes typing {
  0%, 80%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

@media (max-width: 768px) {
  .chat-messages {
    padding: 12px;
  }
  
  .feature-cards {
    grid-template-columns: 1fr;
  }
  
  .message-image {
    max-width: 250px;
  }
  
  .message-video {
    max-width: 100%;
  }
}
</style>
