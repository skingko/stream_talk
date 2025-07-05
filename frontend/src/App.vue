<template>
  <div id="app" class="app-container">


    <!-- 侧边栏 -->
    <div class="sidebar" :class="{ collapsed: sidebarCollapsed }">
      <div class="sidebar-header">
        <div class="logo">
          <div class="logo-placeholder">🎤</div>
          <span v-if="!sidebarCollapsed" class="logo-text">Stream-Omni</span>
        </div>
        <el-button
          @click="toggleSidebar"
          text
          class="collapse-btn"
        >
          {{ sidebarCollapsed ? '展开' : '收起' }}
        </el-button>
      </div>

      <div class="sidebar-content">
        <!-- 新建对话按钮 -->
        <el-button
          type="primary"
          @click="newConversation"
          class="new-chat-btn"
          :class="{ 'icon-only': sidebarCollapsed }"
        >
          <span v-if="!sidebarCollapsed">➕ 新建对话</span>
          <span v-else>➕</span>
        </el-button>
        
        <!-- 对话历史列表 -->
        <div class="conversation-list">
          <div
            v-for="conv in conversations"
            :key="conv.id"
            class="conversation-item"
            :class="{ active: conv.id === currentConversationId }"
            @click="selectConversation(conv.id)"
          >
            <div class="conversation-icon">💬</div>
            <span v-if="!sidebarCollapsed" class="conversation-title">{{ conv.title }}</span>
          </div>
        </div>
      </div>
      
      <!-- 模式切换 -->
      <div class="sidebar-footer">
        <div class="mode-switch">
          <el-segmented 
            v-model="currentMode" 
            :options="modeOptions"
            @change="handleModeChange"
            size="small"
          />
        </div>
        <div v-if="!sidebarCollapsed" class="mode-description">
          {{ getModeDescription() }}
        </div>
      </div>
    </div>

    <!-- 主内容区域 -->
    <div class="main-content">
      <!-- 聊天区域 -->
      <div class="chat-area">
        <ChatMessages
          :messages="currentMessages"
          :loading="isLoading"
          :current-mode="currentMode"
          @stop-generation="stopGeneration"
        />
      </div>

      <!-- 输入区域 -->
      <div class="input-area">
        <!-- 文本模式 -->
        <ChatInput
          v-if="currentMode === 'text'"
          :mode="currentMode"
          :disabled="isLoading"
          @send-message="handleSendMessage"
          @upload-file="handleFileUpload"
          @voice-input="handleVoiceInput"
          @stop-recording="handleStopRecording"
          @stop-speaking="handleStopSpeaking"
        />

        <!-- 语音模式 - 新的智能语音交互界面 -->
        <VoiceConversationInterface
          v-else-if="currentMode === 'voice'"
          @conversation-message="handleConversationMessage"
        />

        <!-- TEN测试模式 -->
        <TenTestInterface
          v-else-if="currentMode === 'test'"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import ChatMessages from './components/ChatMessages.vue'
import ChatInput from './components/ChatInput.vue'
import VoiceConversationInterface from './components/VoiceConversationInterface.vue'
import TenTestInterface from './components/TenTestInterface.vue'
import { useChat } from './composables/useChat'
import { useVoice } from './composables/useVoice'

// 响应式数据
const sidebarCollapsed = ref(false)
const currentMode = ref('text') // 'text' | 'voice'
const isLoading = ref(false)
const currentConversationId = ref(null)

// 模式选项
const modeOptions = [
  { label: '💬', value: 'text' },
  { label: '🎤', value: 'voice' },
  { label: '🔧', value: 'test' }
]

// 对话数据
const conversations = ref([
  { id: 1, title: '新对话', messages: [] }
])

// 使用组合式函数
const { sendMessage, stopGeneration } = useChat()
const { voiceState, initVoice, startRecording, stopRecording, stopSpeaking, toggleMute } = useVoice()

// 计算属性
const currentMessages = computed(() => {
  const conv = conversations.value.find(c => c.id === currentConversationId.value)
  return conv ? conv.messages : []
})

// 方法
const toggleSidebar = () => {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

const newConversation = () => {
  const newId = Date.now()
  conversations.value.unshift({
    id: newId,
    title: '新对话',
    messages: []
  })
  currentConversationId.value = newId
}

const selectConversation = (id) => {
  currentConversationId.value = id
}

const handleModeChange = (mode) => {
  currentMode.value = mode
  console.log('模式切换到:', mode)
}

const getModeDescription = () => {
  switch (currentMode.value) {
    case 'text':
      return '文本输入模式'
    case 'voice':
      return '智能语音对话'
    case 'test':
      return 'TEN框架测试'
    default:
      return '未知模式'
  }
}

const handleSendMessage = async (message) => {
  if (!currentConversationId.value) {
    newConversation()
  }
  
  isLoading.value = true
  try {
    const conv = conversations.value.find(c => c.id === currentConversationId.value)
    if (conv) {
      // 添加用户消息
      conv.messages.push({
        id: Date.now(),
        role: 'user',
        content: message.content,
        type: message.type,
        timestamp: new Date()
      })
      
      // 发送到后端并获取回复
      const response = await sendMessage(message, currentMode.value)
      
      // 添加助手回复
      conv.messages.push({
        id: Date.now() + 1,
        role: 'assistant',
        content: response.content,
        type: response.type,
        timestamp: new Date()
      })
      
      // 更新对话标题
      if (conv.messages.length === 2 && conv.title === '新对话') {
        conv.title = message.content.substring(0, 20) + '...'
      }
    }
  } catch (error) {
    console.error('发送消息失败:', error)
  } finally {
    isLoading.value = false
  }
}

const handleFileUpload = async (file) => {
  console.log('文件上传:', file)
  // 处理文件上传逻辑
}

const handleVoiceInput = async (audioData) => {
  console.log('语音输入:', audioData)
  // 语音输入已在useVoice中处理
}

const handleStopRecording = () => {
  stopRecording()
}

const handleStopSpeaking = () => {
  stopSpeaking()
}

const handleConversationMessage = (message) => {
  // 处理来自语音对话界面的消息
  console.log('语音对话消息:', message)

  if (!currentConversationId.value) {
    newConversation()
  }

  const conv = conversations.value.find(c => c.id === currentConversationId.value)
  if (conv) {
    // 添加消息到对话历史
    conv.messages.push({
      id: Date.now(),
      role: message.role,
      content: message.content,
      type: 'voice',
      timestamp: new Date()
    })

    // 更新对话标题
    if (conv.messages.length === 2 && conv.title === '新对话') {
      conv.title = message.content.substring(0, 20) + '...'
    }
  }
}

// 生命周期
onMounted(async () => {
  // 初始化第一个对话
  if (conversations.value.length > 0) {
    currentConversationId.value = conversations.value[0].id
  }

  // 初始化语音功能
  await initVoice()
})
</script>

<style lang="scss" scoped>
.app-container {
  display: flex;
  height: 100vh;
  background: #f7f7f8;
}

.sidebar {
  width: 260px;
  background: #171717;
  color: white;
  display: flex;
  flex-direction: column;
  transition: width 0.3s ease;
  
  &.collapsed {
    width: 60px;
  }
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #2d2d2d;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
}

.logo-placeholder {
  width: 24px;
  height: 24px;
  font-size: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-img {
  width: 24px;
  height: 24px;
}

.logo-text {
  font-weight: 600;
  font-size: 16px;
}

.collapse-btn {
  color: #8e8ea0;
  
  &:hover {
    color: white;
  }
}

.sidebar-content {
  flex: 1;
  padding: 16px;
  overflow-y: auto;
}

.new-chat-btn {
  width: 100%;
  margin-bottom: 16px;
  
  &.icon-only {
    width: 40px;
    height: 40px;
    padding: 0;
  }
}

.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.conversation-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background: #2d2d2d;
  }
  
  &.active {
    background: #343541;
  }
}

.conversation-icon {
  flex-shrink: 0;
}

.conversation-title {
  flex: 1;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid #2d2d2d;
}

.mode-switch {
  margin-bottom: 8px;
}

.mode-description {
  font-size: 12px;
  color: #8e8ea0;
  text-align: center;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-area {
  flex: 1;
  overflow: hidden;
}

.input-area {
  flex-shrink: 0;
  padding: 16px;
  background: white;
  border-top: 1px solid #e5e5e5;
}
</style>
