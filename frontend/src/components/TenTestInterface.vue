<template>
  <div class="ten-test-interface">
    <div class="header">
      <h2>🎯 TEN框架集成测试</h2>
      <div class="status-indicators">
        <div class="status-item" :class="{ active: backendConnected }">
          <el-icon><Connection /></el-icon>
          <span>后端连接</span>
        </div>
        <div class="status-item" :class="{ active: wsConnected }">
          <el-icon><Microphone /></el-icon>
          <span>WebSocket</span>
        </div>
        <div class="status-item" :class="{ active: tenAvailable }">
          <el-icon><Setting /></el-icon>
          <span>TEN组件</span>
        </div>
      </div>
    </div>

    <!-- 控制面板 -->
    <div class="control-panel">
      <div class="test-section">
        <h3>🔧 连接测试</h3>
        <div class="button-group">
          <el-button @click="testBackendConnection" :loading="testing.backend">
            测试后端连接
          </el-button>
          <el-button @click="testWebSocketConnection" :loading="testing.websocket">
            测试WebSocket
          </el-button>
          <el-button @click="testTenComponents" :loading="testing.ten">
            测试TEN组件
          </el-button>
        </div>
      </div>

      <div class="test-section">
        <h3>🎤 语音测试</h3>
        <div class="voice-controls">
          <el-button 
            type="primary" 
            @click="startVoiceTest" 
            :disabled="!canStartVoice"
            :loading="voiceTest.recording"
          >
            {{ voiceTest.recording ? '录音中...' : '开始语音测试' }}
          </el-button>
          <el-button 
            v-if="voiceTest.recording" 
            type="danger" 
            @click="stopVoiceTest"
          >
            停止录音
          </el-button>
        </div>
        
        <!-- VAD结果显示 -->
        <div v-if="vadResult" class="vad-result">
          <h4>VAD检测结果:</h4>
          <div class="result-grid">
            <div class="result-item">
              <span class="label">语音检测:</span>
              <span class="value" :class="{ active: vadResult.speech_detected }">
                {{ vadResult.speech_detected ? '✅ 检测到语音' : '❌ 未检测到语音' }}
              </span>
            </div>
            <div class="result-item">
              <span class="label">语音比例:</span>
              <span class="value">{{ (vadResult.speech_ratio * 100).toFixed(1) }}%</span>
            </div>
            <div class="result-item">
              <span class="label">总帧数:</span>
              <span class="value">{{ vadResult.total_frames }}</span>
            </div>
            <div class="result-item">
              <span class="label">语音帧:</span>
              <span class="value">{{ vadResult.speech_frames }}</span>
            </div>
          </div>
        </div>
      </div>

      <div class="test-section">
        <h3>💬 对话测试</h3>
        <div class="chat-test">
          <el-input
            v-model="testMessage"
            placeholder="输入测试消息..."
            @keyup.enter="sendTestMessage"
          />
          <el-button @click="sendTestMessage" :loading="testing.chat">
            发送消息
          </el-button>
        </div>
        
        <!-- 对话历史 -->
        <div v-if="chatHistory.length > 0" class="chat-history">
          <h4>对话历史:</h4>
          <div class="message-list">
            <div 
              v-for="(msg, index) in chatHistory" 
              :key="index" 
              class="message-item"
              :class="msg.role"
            >
              <div class="message-role">{{ msg.role === 'user' ? '用户' : 'AI' }}</div>
              <div class="message-content">{{ msg.content }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 日志显示 -->
    <div class="log-section">
      <h3>📋 测试日志</h3>
      <div class="log-container">
        <div 
          v-for="(log, index) in logs" 
          :key="index" 
          class="log-item"
          :class="log.type"
        >
          <span class="timestamp">{{ formatTime(log.timestamp) }}</span>
          <span class="message">{{ log.message }}</span>
        </div>
      </div>
      <div class="log-controls">
        <el-button size="small" @click="clearLogs">清空日志</el-button>
        <el-button size="small" @click="exportLogs">导出日志</el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Connection, Microphone, Setting } from '@element-plus/icons-vue'

// 状态管理
const backendConnected = ref(false)
const wsConnected = ref(false)
const tenAvailable = ref(false)

const testing = reactive({
  backend: false,
  websocket: false,
  ten: false,
  chat: false
})

const voiceTest = reactive({
  recording: false,
  mediaRecorder: null,
  audioChunks: []
})

const vadResult = ref(null)
const testMessage = ref('')
const chatHistory = ref([])
const logs = ref([])

// WebSocket连接
let wsConnection = null

// 计算属性
const canStartVoice = computed(() => {
  return backendConnected.value && wsConnected.value && !voiceTest.recording
})

// 方法
const addLog = (message, type = 'info') => {
  logs.value.push({
    timestamp: new Date(),
    message,
    type
  })
  
  // 限制日志数量
  if (logs.value.length > 100) {
    logs.value.shift()
  }
}

const testBackendConnection = async () => {
  testing.backend = true
  addLog('🔍 测试后端连接...', 'info')
  
  try {
    const response = await fetch('http://localhost:8000/health')
    const data = await response.json()
    
    backendConnected.value = true
    tenAvailable.value = data.ten_components?.vad && data.ten_components?.turn_detection
    
    addLog('✅ 后端连接成功', 'success')
    addLog(`TEN组件状态: VAD=${data.ten_components?.vad}, Turn Detection=${data.ten_components?.turn_detection}`, 'info')
    
    ElMessage.success('后端连接成功')
  } catch (error) {
    backendConnected.value = false
    addLog(`❌ 后端连接失败: ${error.message}`, 'error')
    ElMessage.error('后端连接失败')
  } finally {
    testing.backend = false
  }
}

const testWebSocketConnection = async () => {
  testing.websocket = true
  addLog('🔍 测试WebSocket连接...', 'info')
  
  try {
    wsConnection = new WebSocket('ws://localhost:8000/ws/voice')
    
    wsConnection.onopen = () => {
      wsConnected.value = true
      addLog('✅ WebSocket连接成功', 'success')
      ElMessage.success('WebSocket连接成功')
      testing.websocket = false
    }
    
    wsConnection.onmessage = (event) => {
      const message = JSON.parse(event.data)
      addLog(`📨 收到消息: ${message.type}`, 'info')
      
      if (message.type === 'vad_result') {
        addLog(`🎤 VAD结果: 语音=${message.data.speech_detected}, 能量=${message.data.energy}`, 'info')
      }
    }
    
    wsConnection.onerror = (error) => {
      wsConnected.value = false
      addLog(`❌ WebSocket错误: ${error}`, 'error')
      ElMessage.error('WebSocket连接失败')
      testing.websocket = false
    }
    
    wsConnection.onclose = () => {
      wsConnected.value = false
      addLog('🔌 WebSocket连接关闭', 'warning')
    }
    
  } catch (error) {
    wsConnected.value = false
    addLog(`❌ WebSocket连接失败: ${error.message}`, 'error')
    ElMessage.error('WebSocket连接失败')
    testing.websocket = false
  }
}

const testTenComponents = async () => {
  testing.ten = true
  addLog('🔍 测试TEN组件...', 'info')
  
  try {
    const response = await fetch('http://localhost:8000/')
    const data = await response.json()
    
    addLog(`TEN可用性: ${data.ten_available}`, 'info')
    addLog(`系统初始化: ${data.initialized}`, 'info')
    
    if (data.ten_available && data.initialized) {
      addLog('✅ TEN组件测试通过', 'success')
      ElMessage.success('TEN组件正常')
    } else {
      addLog('⚠️ TEN组件未完全可用', 'warning')
      ElMessage.warning('TEN组件未完全可用')
    }
  } catch (error) {
    addLog(`❌ TEN组件测试失败: ${error.message}`, 'error')
    ElMessage.error('TEN组件测试失败')
  } finally {
    testing.ten = false
  }
}

const startVoiceTest = async () => {
  try {
    addLog('🎤 开始语音测试...', 'info')
    
    // 获取麦克风权限
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    
    voiceTest.mediaRecorder = new MediaRecorder(stream)
    voiceTest.audioChunks = []
    
    voiceTest.mediaRecorder.ondataavailable = (event) => {
      voiceTest.audioChunks.push(event.data)
    }
    
    voiceTest.mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(voiceTest.audioChunks, { type: 'audio/wav' })
      await processVoiceData(audioBlob)
    }
    
    voiceTest.mediaRecorder.start()
    voiceTest.recording = true
    
    addLog('✅ 录音开始', 'success')
    ElMessage.success('录音开始，请说话...')
    
  } catch (error) {
    addLog(`❌ 语音测试失败: ${error.message}`, 'error')
    ElMessage.error('无法访问麦克风')
  }
}

const stopVoiceTest = () => {
  if (voiceTest.mediaRecorder && voiceTest.recording) {
    voiceTest.mediaRecorder.stop()
    voiceTest.recording = false
    addLog('🛑 录音停止', 'info')
  }
}

const processVoiceData = async (audioBlob) => {
  try {
    addLog('🔄 处理语音数据...', 'info')
    
    const formData = new FormData()
    formData.append('audio_file', audioBlob, 'test.wav')
    
    const response = await fetch('http://localhost:8000/api/voice/process', {
      method: 'POST',
      body: formData
    })
    
    const result = await response.json()
    vadResult.value = result.vad_results
    
    addLog('✅ 语音处理完成', 'success')
    addLog(`转录结果: ${result.transcript}`, 'info')
    
    ElMessage.success('语音处理完成')
    
  } catch (error) {
    addLog(`❌ 语音处理失败: ${error.message}`, 'error')
    ElMessage.error('语音处理失败')
  }
}

const sendTestMessage = async () => {
  if (!testMessage.value.trim()) return
  
  testing.chat = true
  addLog(`📤 发送消息: ${testMessage.value}`, 'info')
  
  try {
    chatHistory.value.push({
      role: 'user',
      content: testMessage.value
    })
    
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        messages: [{ role: 'user', content: testMessage.value }],
        mode: 'text',
        stream: false
      })
    })
    
    const result = await response.json()
    const aiResponse = result.choices[0].message.content
    
    chatHistory.value.push({
      role: 'assistant',
      content: aiResponse
    })
    
    addLog(`📥 收到回复: ${aiResponse}`, 'success')
    testMessage.value = ''
    
  } catch (error) {
    addLog(`❌ 发送消息失败: ${error.message}`, 'error')
    ElMessage.error('发送消息失败')
  } finally {
    testing.chat = false
  }
}

const clearLogs = () => {
  logs.value = []
  addLog('📋 日志已清空', 'info')
}

const exportLogs = () => {
  const logText = logs.value.map(log => 
    `[${formatTime(log.timestamp)}] ${log.type.toUpperCase()}: ${log.message}`
  ).join('\n')
  
  const blob = new Blob([logText], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `ten-test-logs-${new Date().toISOString().slice(0, 19)}.txt`
  a.click()
  URL.revokeObjectURL(url)
}

const formatTime = (date) => {
  return date.toLocaleTimeString()
}

// 生命周期
onMounted(() => {
  addLog('🚀 TEN测试界面已加载', 'info')
  // 自动测试后端连接
  testBackendConnection()
})

onUnmounted(() => {
  if (wsConnection) {
    wsConnection.close()
  }
  if (voiceTest.mediaRecorder && voiceTest.recording) {
    voiceTest.mediaRecorder.stop()
  }
})
</script>

<style scoped>
.ten-test-interface {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 2px solid #e0e0e0;
}

.status-indicators {
  display: flex;
  gap: 20px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: 20px;
  background: #f5f5f5;
  color: #666;
  transition: all 0.3s;
}

.status-item.active {
  background: #e8f5e8;
  color: #52c41a;
}

.control-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
  margin-bottom: 30px;
}

.test-section {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.test-section h3 {
  margin: 0 0 15px 0;
  color: #333;
}

.button-group {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.voice-controls {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.vad-result {
  background: #f8f9fa;
  padding: 15px;
  border-radius: 8px;
  border-left: 4px solid #1890ff;
}

.result-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 10px;
}

.result-item {
  display: flex;
  justify-content: space-between;
}

.result-item .label {
  font-weight: 500;
  color: #666;
}

.result-item .value {
  font-weight: 600;
}

.result-item .value.active {
  color: #52c41a;
}

.chat-test {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.chat-history {
  max-height: 300px;
  overflow-y: auto;
}

.message-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.message-item {
  padding: 10px;
  border-radius: 8px;
  background: #f5f5f5;
}

.message-item.user {
  background: #e6f7ff;
  margin-left: 20px;
}

.message-item.assistant {
  background: #f6ffed;
  margin-right: 20px;
}

.message-role {
  font-size: 12px;
  font-weight: 600;
  color: #666;
  margin-bottom: 5px;
}

.log-section {
  background: white;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.log-container {
  height: 300px;
  overflow-y: auto;
  background: #1e1e1e;
  color: #fff;
  padding: 15px;
  border-radius: 8px;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  margin-bottom: 15px;
}

.log-item {
  margin-bottom: 5px;
  display: flex;
  gap: 10px;
}

.log-item.error {
  color: #ff6b6b;
}

.log-item.success {
  color: #51cf66;
}

.log-item.warning {
  color: #ffd43b;
}

.log-item.info {
  color: #74c0fc;
}

.timestamp {
  color: #868e96;
  min-width: 80px;
}

.log-controls {
  display: flex;
  gap: 10px;
}
</style>
