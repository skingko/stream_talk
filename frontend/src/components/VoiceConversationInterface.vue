<template>
  <div class="voice-conversation-interface">
    <!-- 主要可视化区域 -->
    <div class="conversation-visualizer">
      <!-- 中央圆形可视化 -->
      <div class="central-visualizer" :class="visualizationClass">
        <div class="outer-ring" :class="{ active: isActive }">
          <div class="middle-ring">
            <div class="inner-core">
              <!-- 波纹动画 -->
              <div v-if="showWaves" class="wave-container">
                <div 
                  v-for="i in 4" 
                  :key="i" 
                  class="wave" 
                  :style="{ animationDelay: (i - 1) * 0.2 + 's' }"
                ></div>
              </div>
              
              <!-- 状态图标 -->
              <div class="state-icon">
                <el-icon v-if="currentState === 'idle'" size="48"><Microphone /></el-icon>
                <el-icon v-else-if="currentState === 'listening'" size="48"><Microphone /></el-icon>
                <el-icon v-else-if="currentState === 'recording'" size="48"><VideoPause /></el-icon>
                <el-icon v-else-if="currentState === 'processing'" size="48"><Loading /></el-icon>
                <el-icon v-else-if="currentState === 'speaking'" size="48"><Mute /></el-icon>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 状态文本 -->
      <div class="state-description">
        {{ stateDescription }}
      </div>
      
      <!-- 实时转录显示 -->
      <div v-if="showTranscript" class="transcript-display">
        <div class="transcript-label">实时转录:</div>
        <div class="transcript-text">
          {{ realtimeTranscript || finalTranscript || '等待语音输入...' }}
        </div>
      </div>
    </div>
    
    <!-- 控制面板 -->
    <div class="control-panel">
      <!-- 主要控制按钮 -->
      <div class="main-controls">
        <!-- 开始/停止对话按钮 -->
        <el-button
          v-if="!isVoiceMode"
          type="primary"
          size="large"
          :icon="Microphone"
          @click="startConversation"
          :loading="isInitializing"
          class="start-btn"
        >
          开始语音对话
        </el-button>
        
        <el-button
          v-else
          type="danger"
          size="large"
          :icon="Close"
          @click="stopConversation"
          class="stop-btn"
        >
          结束对话
        </el-button>
        
        <!-- 中断按钮 -->
        <el-button
          v-if="isSpeaking"
          type="warning"
          size="large"
          :icon="VideoPause"
          @click="interruptResponse"
          class="interrupt-btn"
        >
          中断回复
        </el-button>
      </div>
      
      <!-- 辅助控制 -->
      <div class="auxiliary-controls">
        <!-- 音量控制 -->
        <div class="volume-control">
          <el-icon><Mute /></el-icon>
          <el-slider
            v-model="volume"
            :min="0"
            :max="100"
            @change="handleVolumeChange"
            style="width: 100px; margin: 0 8px;"
          />
        </div>
        
        <!-- 设置按钮 -->
        <el-button
          :icon="Setting"
          circle
          @click="showSettings = true"
          class="settings-btn"
        />
      </div>
    </div>
    
    <!-- 对话统计信息 -->
    <div v-if="showStats" class="conversation-stats">
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-value">{{ conversationState.turnCount }}</div>
          <div class="stat-label">对话轮次</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{{ conversationState.interruptionCount }}</div>
          <div class="stat-label">中断次数</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">{{ formatDuration(conversationDuration) }}</div>
          <div class="stat-label">对话时长</div>
        </div>
      </div>
    </div>
    
    <!-- 错误提示 -->
    <div v-if="error" class="error-display">
      <el-alert
        :title="error"
        type="error"
        :closable="true"
        @close="clearError"
        show-icon
      />
    </div>
    
    <!-- 设置对话框 -->
    <el-dialog
      v-model="showSettings"
      title="语音对话设置"
      width="500px"
    >
      <div class="settings-content">
        <!-- VAD设置 -->
        <div class="setting-group">
          <h4>语音检测设置</h4>
          <el-form label-width="120px">
            <el-form-item label="检测敏感度:">
              <el-slider
                v-model="vadSensitivity"
                :min="0.1"
                :max="1.0"
                :step="0.1"
                show-input
              />
            </el-form-item>
            <el-form-item label="中断阈值:">
              <el-input-number
                v-model="interruptionThreshold"
                :min="100"
                :max="1000"
                :step="50"
                controls-position="right"
              />
              <span style="margin-left: 8px; color: #666;">毫秒</span>
            </el-form-item>
          </el-form>
        </div>
        
        <!-- 显示设置 -->
        <div class="setting-group">
          <h4>显示设置</h4>
          <el-form label-width="120px">
            <el-form-item label="显示转录:">
              <el-switch v-model="showTranscript" />
            </el-form-item>
            <el-form-item label="显示统计:">
              <el-switch v-model="showStats" />
            </el-form-item>
          </el-form>
        </div>
      </div>
      
      <template #footer>
        <el-button @click="showSettings = false">取消</el-button>
        <el-button type="primary" @click="saveSettings">保存设置</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Microphone, VideoPause, Close, Loading,
  Mute, Setting
} from '@element-plus/icons-vue'
import { useVoiceConversation } from '../composables/useVoiceConversation'

// 使用语音对话功能
const {
  conversationState,
  getStateDescription,
  getVisualizationState,
  initVoiceConversation,
  startVoiceConversation,
  stopVoiceConversation,
  interruptAIResponse,
  cleanup
} = useVoiceConversation()

// 响应式数据
const isInitializing = ref(false)
const showSettings = ref(false)
const showTranscript = ref(true)
const showStats = ref(false)
const volume = ref(80)
const vadSensitivity = ref(0.7)
const interruptionThreshold = ref(300)
const conversationStartTime = ref(null)

// 计算属性
const isVoiceMode = computed(() => conversationState.isVoiceMode)
const currentState = computed(() => conversationState.currentState)
const isListening = computed(() => conversationState.isListening)
const isProcessing = computed(() => conversationState.isProcessing)
const isSpeaking = computed(() => conversationState.isSpeaking)
const realtimeTranscript = computed(() => conversationState.realtimeTranscript)
const finalTranscript = computed(() => conversationState.finalTranscript)
const error = computed(() => conversationState.error)

const stateDescription = computed(() => getStateDescription.value)

const visualizationClass = computed(() => {
  const state = getVisualizationState.value
  return {
    'state-idle': state === 'idle',
    'state-listening': state === 'listening',
    'state-recording': state === 'recording',
    'state-speaking': state === 'speaking'
  }
})

const isActive = computed(() => {
  return isListening.value || isProcessing.value || isSpeaking.value
})

const showWaves = computed(() => {
  return conversationState.speechDetected || isSpeaking.value
})

const conversationDuration = computed(() => {
  if (!conversationStartTime.value) return 0
  return Date.now() - conversationStartTime.value
})

// 方法
const startConversation = async () => {
  try {
    isInitializing.value = true
    
    // 初始化语音对话系统
    const initialized = await initVoiceConversation()
    if (!initialized) {
      throw new Error('语音对话系统初始化失败')
    }
    
    // 开始语音对话
    await startVoiceConversation()
    conversationStartTime.value = Date.now()
    
  } catch (error) {
    console.error('❌ 开始对话失败:', error)
    ElMessage.error(`开始对话失败: ${error.message}`)
  } finally {
    isInitializing.value = false
  }
}

const stopConversation = () => {
  stopVoiceConversation()
  conversationStartTime.value = null
}

const interruptResponse = () => {
  interruptAIResponse()
}

const handleVolumeChange = (value) => {
  // 调整音量
  console.log('音量调整到:', value)
}

const clearError = () => {
  conversationState.error = null
}

const saveSettings = () => {
  // 保存设置到localStorage
  const settings = {
    showTranscript: showTranscript.value,
    showStats: showStats.value,
    volume: volume.value,
    vadSensitivity: vadSensitivity.value,
    interruptionThreshold: interruptionThreshold.value
  }
  
  localStorage.setItem('voiceConversationSettings', JSON.stringify(settings))
  
  // 应用设置
  conversationState.interruptionThreshold = interruptionThreshold.value
  
  showSettings.value = false
  ElMessage.success('设置已保存')
}

const loadSettings = () => {
  try {
    const saved = localStorage.getItem('voiceConversationSettings')
    if (saved) {
      const settings = JSON.parse(saved)
      showTranscript.value = settings.showTranscript ?? true
      showStats.value = settings.showStats ?? false
      volume.value = settings.volume ?? 80
      vadSensitivity.value = settings.vadSensitivity ?? 0.7
      interruptionThreshold.value = settings.interruptionThreshold ?? 300
    }
  } catch (error) {
    console.error('加载设置失败:', error)
  }
}

const formatDuration = (ms) => {
  if (!ms) return '00:00'
  const seconds = Math.floor(ms / 1000)
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`
}

// 生命周期
onMounted(() => {
  loadSettings()
})

onUnmounted(() => {
  cleanup()
})

// 监听状态变化
watch(currentState, (newState, oldState) => {
  console.log(`状态变化: ${oldState} -> ${newState}`)
})
</script>

<style lang="scss" scoped>
.voice-conversation-interface {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 32px;
  min-height: 600px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 16px;
}

.conversation-visualizer {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 32px;
}

.central-visualizer {
  position: relative;
  width: 280px;
  height: 280px;
  margin-bottom: 24px;
}

.outer-ring {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  position: relative;
  
  &.active {
    transform: scale(1.05);
    box-shadow: 0 0 40px rgba(102, 126, 234, 0.4);
  }
  
  &::before {
    content: '';
    position: absolute;
    top: -10px;
    left: -10px;
    right: -10px;
    bottom: -10px;
    border-radius: 50%;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    opacity: 0.3;
    z-index: -1;
    animation: pulse 2s infinite;
  }
}

.middle-ring {
  width: 220px;
  height: 220px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
}

.inner-core {
  width: 160px;
  height: 160px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.wave-container {
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

.state-icon {
  color: white;
  z-index: 10;
}

// 状态样式
.state-idle .outer-ring {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.state-listening .outer-ring {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
  animation: breathe 3s infinite;
}

.state-recording .outer-ring {
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
  animation: pulse 1s infinite;
}

.state-speaking .outer-ring {
  background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
  animation: speak 1.5s infinite;
}

.state-description {
  font-size: 20px;
  font-weight: 600;
  color: #2c3e50;
  text-align: center;
  margin-bottom: 16px;
}

.transcript-display {
  background: rgba(255, 255, 255, 0.9);
  padding: 20px;
  border-radius: 12px;
  min-width: 400px;
  max-width: 600px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  
  .transcript-label {
    font-size: 14px;
    color: #666;
    margin-bottom: 8px;
  }
  
  .transcript-text {
    font-size: 16px;
    color: #2c3e50;
    line-height: 1.5;
    min-height: 24px;
  }
}

.control-panel {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  width: 100%;
  max-width: 500px;
}

.main-controls {
  display: flex;
  gap: 16px;
  align-items: center;
}

.start-btn, .stop-btn, .interrupt-btn {
  height: 56px;
  padding: 0 32px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 28px;
}

.auxiliary-controls {
  display: flex;
  align-items: center;
  gap: 20px;
}

.volume-control {
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.9);
  padding: 8px 16px;
  border-radius: 20px;
}

.settings-btn {
  background: rgba(255, 255, 255, 0.9);
}

.conversation-stats {
  margin-top: 24px;
  background: rgba(255, 255, 255, 0.9);
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.stat-item {
  text-align: center;
  
  .stat-value {
    font-size: 24px;
    font-weight: 700;
    color: #2c3e50;
  }
  
  .stat-label {
    font-size: 12px;
    color: #666;
    margin-top: 4px;
  }
}

.error-display {
  margin-top: 20px;
  width: 100%;
  max-width: 500px;
}

.settings-content {
  .setting-group {
    margin-bottom: 24px;
    
    h4 {
      margin-bottom: 16px;
      color: #2c3e50;
    }
  }
}

// 动画
@keyframes pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.7;
  }
}

@keyframes breathe {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.03);
    opacity: 0.8;
  }
}

@keyframes speak {
  0%, 100% {
    transform: scale(1);
  }
  25% {
    transform: scale(1.02);
  }
  75% {
    transform: scale(1.04);
  }
}

@keyframes wave {
  0% {
    transform: scale(1);
    opacity: 1;
  }
  100% {
    transform: scale(1.6);
    opacity: 0;
  }
}

// 响应式设计
@media (max-width: 768px) {
  .voice-conversation-interface {
    padding: 20px;
  }
  
  .central-visualizer {
    width: 220px;
    height: 220px;
  }
  
  .outer-ring {
    width: 220px;
    height: 220px;
  }
  
  .middle-ring {
    width: 170px;
    height: 170px;
  }
  
  .inner-core {
    width: 120px;
    height: 120px;
  }
  
  .transcript-display {
    min-width: 300px;
    max-width: 400px;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
    gap: 12px;
  }
}
</style>
