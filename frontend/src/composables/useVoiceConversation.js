import { ref, reactive, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'

// 对话状态枚举
const ConversationState = {
  IDLE: 'idle',                    // 空闲状态
  LISTENING: 'listening',          // 监听用户
  RECORDING: 'recording',          // 录音中
  PROCESSING: 'processing',        // 处理用户输入
  SPEAKING: 'speaking',           // AI回复中
  INTERRUPTED: 'interrupted',      // 被中断
  WAITING: 'waiting'              // 等待用户继续
}

// 全局对话状态
const conversationState = reactive({
  currentState: ConversationState.IDLE,
  isVoiceMode: false,
  isListening: false,
  isProcessing: false,
  isSpeaking: false,
  isInterrupted: false,
  
  // VAD相关
  vadEnabled: false,
  speechDetected: false,
  silenceDetected: false,
  
  // 转录相关
  realtimeTranscript: '',
  finalTranscript: '',
  
  // 对话管理
  conversationId: null,
  turnCount: 0,
  lastUserSpeechTime: null,
  lastAISpeechTime: null,
  messages: [], // 对话消息历史
  
  // 中断处理
  interruptionCount: 0,
  interruptionThreshold: 300, // 300ms内检测到语音即中断
  
  // 错误处理
  error: null,
  
  // 统计信息
  stats: {
    totalConversations: 0,
    totalInterruptions: 0,
    successfulInterruptions: 0,
    averageResponseTime: 0
  }
})

// WebSocket连接
let wsConnection = null
let reconnectAttempts = 0
const maxReconnectAttempts = 5
let heartbeatInterval = null
let connectionCheckInterval = null

// 音频相关
let audioContext = null
let mediaRecorder = null
let audioStream = null
let vadProcessor = null

export function useVoiceConversation() {
  
  /**
   * 初始化语音对话系统
   */
  const initVoiceConversation = async () => {
    try {
      console.log('🚀 初始化语音对话系统...')
      
      // 1. 初始化WebSocket连接
      await initWebSocketConnection()
      
      // 2. 初始化音频系统
      await initAudioSystem()
      
      // 3. 初始化VAD
      await initVAD()
      
      console.log('✅ 语音对话系统初始化完成')
      return true
      
    } catch (error) {
      console.error('❌ 语音对话系统初始化失败:', error)
      conversationState.error = error.message
      return false
    }
  }
  
  /**
   * 初始化WebSocket连接
   */
  const initWebSocketConnection = async () => {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = `ws://localhost:8002/ws/voice`
        wsConnection = new WebSocket(wsUrl)
        
        wsConnection.onopen = () => {
          console.log('🔗 WebSocket连接已建立')
          conversationState.error = null
          reconnectAttempts = 0

          // 发送初始化消息
          wsConnection.send(JSON.stringify({
            type: 'init',
            data: { client_type: 'vue_frontend' }
          }))

          // 启动心跳机制
          startHeartbeat()

          // 启动连接状态检查
          startConnectionCheck()

          resolve()
        }
        
        wsConnection.onmessage = (event) => {
          handleWebSocketMessage(JSON.parse(event.data))
        }
        
        wsConnection.onclose = (event) => {
          console.log('🔌 WebSocket连接已关闭:', event.code, event.reason)

          // 清理定时器
          stopHeartbeat()
          stopConnectionCheck()

          // 在语音模式下自动重连，无论是否正常关闭
          if (conversationState.isVoiceMode && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++
            console.log(`🔄 检测到连接断开，自动重连 (${reconnectAttempts}/${maxReconnectAttempts})...`)

            // 显示重连提示
            ElMessage.warning(`连接断开，正在重连... (${reconnectAttempts}/${maxReconnectAttempts})`)

            const reconnectDelay = 1000 // 1秒后重连
            setTimeout(() => {
              initWebSocketConnection().then(() => {
                // 重连成功后恢复语音监听状态
                console.log('🔄 重连成功，恢复语音监听状态')
                conversationState.currentState = ConversationState.LISTENING
                conversationState.vadEnabled = true
                ElMessage.success('连接已恢复，可以继续语音对话')
                reconnectAttempts = 0 // 重置重连次数
              }).catch(error => {
                console.error('❌ 重连失败:', error)
                if (reconnectAttempts >= maxReconnectAttempts) {
                  ElMessage.error('连接断开，请手动重新开始对话')
                  conversationState.currentState = ConversationState.IDLE
                  conversationState.isVoiceMode = false
                }
              })
            }, reconnectDelay)
          } else if (!conversationState.isVoiceMode) {
            reconnectAttempts = 0
          }
        }
        
        wsConnection.onerror = (error) => {
          console.error('❌ WebSocket错误:', error)
          reject(error)
        }
        
        // 超时处理
        setTimeout(() => {
          if (wsConnection.readyState !== WebSocket.OPEN) {
            reject(new Error('WebSocket连接超时'))
          }
        }, 10000)
        
      } catch (error) {
        reject(error)
      }
    })
  }
  
  /**
   * 启动心跳机制
   */
  const startHeartbeat = () => {
    // 每10秒发送一次心跳，更频繁以防止超时
    heartbeatInterval = setInterval(() => {
      if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({
          type: 'ping',
          data: { timestamp: Date.now() }
        }))
        console.log('💓 发送心跳')
      }
    }, 10000)
  }

  /**
   * 停止心跳机制
   */
  const stopHeartbeat = () => {
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval)
      heartbeatInterval = null
      console.log('💔 停止心跳')
    }
  }

  /**
   * 启动连接状态检查
   */
  const startConnectionCheck = () => {
    // 每5秒检查一次连接状态
    connectionCheckInterval = setInterval(() => {
      if (wsConnection && wsConnection.readyState !== WebSocket.OPEN) {
        console.warn('⚠️ WebSocket连接异常，状态:', wsConnection.readyState)

        // 如果连接断开且正在进行对话，尝试重连
        if (conversationState.isVoiceMode) {
          console.log('🔄 检测到连接断开，尝试重连...')
          initWebSocketConnection()
        }
      }
    }, 5000)
  }

  /**
   * 停止连接状态检查
   */
  const stopConnectionCheck = () => {
    if (connectionCheckInterval) {
      clearInterval(connectionCheckInterval)
      connectionCheckInterval = null
      console.log('🛑 停止连接检查')
    }
  }

  /**
   * 处理WebSocket消息
   */
  const handleWebSocketMessage = (message) => {
    const { type, data } = message
    
    switch (type) {
      case 'status':
        console.log('📊 服务器状态:', data)
        break

      case 'conversation_started':
        console.log('🎯 对话已开始:', data)
        conversationState.conversationId = data.conversation_id
        conversationState.currentState = ConversationState.LISTENING
        break

      case 'conversation_stopped':
        console.log('🛑 对话已停止:', data)
        conversationState.conversationId = null
        conversationState.currentState = ConversationState.IDLE
        break

      case 'vad_result':
        handleVADResult(data)
        break

      case 'vad_event':
        handleVADEvent(data)
        break

      case 'turn_detection_event':
        handleTurnDetectionEvent(data)
        break

      case 'asr_result':
        handleASRResult(data)
        break

      case 'llm_response':
        handleLLMResponse(data)
        break

      case 'tts_event':
        handleTTSEvent(data)
        break

      case 'tts_audio':
        handleTTSAudio(data)
        break

      case 'conversation_state_change':
        handleStateChange(data)
        break

      case 'error':
        handleError(data)
        break

      case 'pong':
        console.log('💓 收到心跳响应')
        break

      case 'processing':
        console.log('⏳ 服务器处理中:', data.message)
        break

      // TEN框架专用消息类型
      case 'connection_established':
        console.log('🔌 连接已建立:', data)
        break

      case 'speech_start':
        console.log('🎤 TEN VAD: 检测到语音开始')
        conversationState.currentState = ConversationState.RECORDING
        break

      case 'speech_end':
        console.log('🎤 TEN Turn Detection: 语音结束')
        conversationState.currentState = ConversationState.PROCESSING
        break

      case 'transcription_result':
        console.log('📝 转录结果:', data.text)
        handleTranscriptionResult(data)
        break

      case 'tts_start':
        console.log('🔊 TTS开始播放:', data.text)
        conversationState.currentState = ConversationState.SPEAKING
        // 回音抑制：TTS播放期间禁用VAD
        conversationState.vadEnabled = false
        console.log('🔇 回音抑制：TTS播放期间禁用VAD')
        break

      case 'tts_end':
        console.log('🔊 TTS生成结束，等待音频播放完成')
        conversationState.currentState = ConversationState.LISTENING
        // 注意：不要立即重新启用VAD，等待音频播放完成后再启用
        // VAD将在音频队列播放完成后自动重新启用
        break

      case 'listening_ready':
        console.log('🎤 准备接收下一轮语音输入:', data.message)
        conversationState.currentState = ConversationState.LISTENING
        // 确保VAD处于激活状态
        conversationState.vadEnabled = true
        // 显示提示信息
        ElMessage.info('准备接收下一轮语音输入，请开始说话')
        break

      case 'init_response':
        console.log('🔧 收到初始化响应:', data)
        break

      case 'audio_received':
        console.log('🎵 音频数据已接收:', data)
        break

      case 'audio_chunk':
        console.log('🎵 收到音频块:', data.chunk_id)
        handleAudioChunk(data)
        break

      case 'tts_error':
        console.error('❌ TTS错误:', data.error)
        ElMessage.error(`TTS播放失败: ${data.error}`)
        conversationState.currentState = ConversationState.LISTENING
        break

      default:
        console.warn('未知消息类型:', type, data)
    }
  }

  /**
   * 处理转录结果
   */
  const handleTranscriptionResult = (data) => {
    const { text, confidence } = data

    // 更新转录文本
    conversationState.finalTranscript = text

    // 添加到对话历史
    if (text && text.trim()) {
      conversationState.messages.push({
        id: Date.now(),
        type: 'user',
        content: text,
        timestamp: new Date(),
        confidence: confidence
      })
    }

    console.log('📝 转录完成:', text, '置信度:', confidence)
  }

  // 音频播放队列管理
  let audioQueue = []
  let isPlaying = false
  let nextPlayTime = 0
  let currentGainNode = null

  /**
   * 处理音频块
   */
  const handleAudioChunk = (data) => {
    const { audio, chunk_id, sample_rate } = data

    try {
      // 解码base64音频数据
      const audioBytes = atob(audio)
      const audioArray = new Int16Array(audioBytes.length / 2)

      for (let i = 0; i < audioArray.length; i++) {
        audioArray[i] = (audioBytes.charCodeAt(i * 2) & 0xFF) |
                       ((audioBytes.charCodeAt(i * 2 + 1) & 0xFF) << 8)
      }

      // 添加到音频队列
      audioQueue.push({
        audioData: audioArray,
        sampleRate: sample_rate || 22050,
        chunkId: chunk_id
      })

      // 开始播放队列
      if (!isPlaying) {
        playAudioQueue()
      }

      console.log('🎵 音频块加入队列:', chunk_id, '队列长度:', audioQueue.length)

    } catch (error) {
      console.error('❌ 音频块处理失败:', error)
    }
  }

  /**
   * 播放音频队列
   */
  const playAudioQueue = async () => {
    if (isPlaying || audioQueue.length === 0) {
      return
    }

    isPlaying = true
    console.log('🎵 开始播放音频队列，共', audioQueue.length, '个音频块')

    try {
      while (audioQueue.length > 0) {
        const audioItem = audioQueue.shift()
        await playAudioChunk(audioItem.audioData, audioItem.sampleRate)

        // 添加小间隔防止音频块重叠
        await new Promise(resolve => setTimeout(resolve, 10))
      }
    } catch (error) {
      console.error('❌ 音频队列播放失败:', error)
    } finally {
      isPlaying = false
      console.log('🎵 音频队列播放完成')

      // 音频播放完成后，检查是否需要重新启用VAD
      if (conversationState.currentState === ConversationState.LISTENING) {
        // 延迟重新启用VAD，避免回音
        setTimeout(() => {
          if (conversationState.currentState === ConversationState.LISTENING) {
            conversationState.vadEnabled = true
            console.log('🎤 音频播放完成，重新启用VAD监听')
          }
        }, 1000) // 1秒延迟
      }
    }
  }

  /**
   * 播放单个音频块
   */
  const playAudioChunk = (audioData, sampleRate) => {
    return new Promise((resolve, reject) => {
      try {
        if (!audioContext) {
          console.warn('⚠️ 音频上下文未初始化')
          resolve()
          return
        }

        // 创建音频缓冲区
        const buffer = audioContext.createBuffer(1, audioData.length, sampleRate)
        const channelData = buffer.getChannelData(0)

        // 转换为float32格式并应用音量控制
        for (let i = 0; i < audioData.length; i++) {
          channelData[i] = (audioData[i] / 32768.0) * 0.7  // 降低音量到70%
        }

        // 创建音频源
        const source = audioContext.createBufferSource()
        source.buffer = buffer

        // 创建增益节点用于音量控制
        const gainNode = audioContext.createGain()
        gainNode.gain.setValueAtTime(0.7, audioContext.currentTime)

        // 连接音频节点
        source.connect(gainNode)
        gainNode.connect(audioContext.destination)

        // 计算播放时间，确保音频块按顺序播放
        const currentTime = audioContext.currentTime
        const startTime = Math.max(currentTime, nextPlayTime)
        const duration = buffer.duration

        // 更新下次播放时间
        nextPlayTime = startTime + duration

        // 播放音频
        source.start(startTime)

        // 在音频播放完成后resolve
        source.onended = () => {
          resolve()
        }

        source.onerror = (error) => {
          console.error('❌ 音频源播放错误:', error)
          reject(error)
        }

      } catch (error) {
        console.error('❌ 音频播放失败:', error)
        reject(error)
      }
    })
  }

  /**
   * 停止当前音频播放
   */
  const stopAudioPlayback = () => {
    try {
      // 清空音频队列
      audioQueue = []
      isPlaying = false
      nextPlayTime = 0

      // 停止当前播放的音频
      if (currentGainNode) {
        currentGainNode.gain.setValueAtTime(0, audioContext.currentTime)
        currentGainNode = null
      }

      console.log('🛑 音频播放已停止')
    } catch (error) {
      console.error('❌ 停止音频播放失败:', error)
    }
  }

  /**
   * 初始化音频系统
   */
  const initAudioSystem = async () => {
    try {
      // 请求麦克风权限
      audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000,  // TEN VAD要求16kHz
          channelCount: 1
        }
      })
      
      // 初始化音频上下文
      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      })
      
      // 初始化媒体录制器
      mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 16000
      })
      
      console.log('🎤 音频系统初始化完成')
      
    } catch (error) {
      throw new Error(`音频系统初始化失败: ${error.message}`)
    }
  }
  
  /**
   * 初始化VAD (语音活动检测)
   */
  const initVAD = async () => {
    try {
      // 创建音频处理节点
      const source = audioContext.createMediaStreamSource(audioStream)
      
      // 创建ScriptProcessor用于实时音频处理
      // 注意: ScriptProcessorNode已弃用，但为了兼容性暂时保留
      // TODO: 未来版本将升级到AudioWorkletNode
      vadProcessor = audioContext.createScriptProcessor(1024, 1, 1)
      
      vadProcessor.onaudioprocess = (event) => {
        if (!conversationState.vadEnabled) {
          console.debug('🔇 VAD未启用，跳过音频处理')
          return
        }

        const inputBuffer = event.inputBuffer.getChannelData(0)

        // 计算音频能量用于调试
        const energy = inputBuffer.reduce((sum, sample) => sum + sample * sample, 0) / inputBuffer.length
        if (energy > 0.001) {
          console.debug('🎤 检测到音频活动，能量:', energy.toFixed(6))
        }

        // 发送音频数据到后端VAD
        if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
          // 将Float32Array转换为Int16Array以减少数据量
          const int16Array = new Int16Array(inputBuffer.length)
          for (let i = 0; i < inputBuffer.length; i++) {
            int16Array[i] = Math.max(-32768, Math.min(32767, inputBuffer[i] * 32768))
          }
          
          // 转换为base64编码
          const audioBuffer = new ArrayBuffer(int16Array.length * 2)
          const view = new DataView(audioBuffer)
          for (let i = 0; i < int16Array.length; i++) {
            view.setInt16(i * 2, int16Array[i], true) // little endian
          }
          const base64Audio = btoa(String.fromCharCode(...new Uint8Array(audioBuffer)))

          const audioMessage = {
            type: 'audio_data',
            data: {
              audio: base64Audio,
              sample_rate: 16000,
              timestamp: Date.now(),
              conversation_id: conversationState.conversationId
            }
          }

          wsConnection.send(JSON.stringify(audioMessage))
          console.debug('🎵 发送音频数据，长度:', base64Audio.length, '字节')
        }
      }
      
      source.connect(vadProcessor)
      vadProcessor.connect(audioContext.destination)
      
      console.log('🔊 VAD初始化完成')
      
    } catch (error) {
      throw new Error(`VAD初始化失败: ${error.message}`)
    }
  }
  
  /**
   * 开始语音对话
   */
  const startVoiceConversation = async () => {
    try {
      if (!wsConnection || wsConnection.readyState !== WebSocket.OPEN) {
        throw new Error('WebSocket连接未建立')
      }
      
      // 生成新的对话ID
      conversationState.conversationId = `conv_${Date.now()}`
      conversationState.turnCount = 0
      conversationState.isVoiceMode = true
      
      // 启动VAD监听
      conversationState.vadEnabled = true
      
      // 发送开始对话命令
      wsConnection.send(JSON.stringify({
        type: 'start_conversation',
        data: {
          conversation_id: conversationState.conversationId,
          mode: 'voice'
        }
      }))
      
      // 更新状态
      updateConversationState(ConversationState.LISTENING)
      
      console.log('🎙️ 开始语音对话:', conversationState.conversationId)
      
      // 更新统计
      conversationState.stats.totalConversations++
      
      ElMessage.success('语音对话已开始，请开始说话')
      
    } catch (error) {
      console.error('❌ 开始语音对话失败:', error)
      conversationState.error = error.message
      ElMessage.error(`开始对话失败: ${error.message}`)
    }
  }
  
  /**
   * 停止语音对话
   */
  const stopVoiceConversation = () => {
    try {
      // 停止VAD
      conversationState.vadEnabled = false
      
      // 发送停止对话命令
      if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
        wsConnection.send(JSON.stringify({
          type: 'stop_conversation',
          data: {
            conversation_id: conversationState.conversationId
          }
        }))
      }
      
      // 重置状态
      conversationState.isVoiceMode = false
      conversationState.conversationId = null
      conversationState.realtimeTranscript = ''
      conversationState.finalTranscript = ''
      
      updateConversationState(ConversationState.IDLE)
      
      console.log('🛑 停止语音对话')
      ElMessage.info('语音对话已停止')
      
    } catch (error) {
      console.error('❌ 停止语音对话失败:', error)
    }
  }
  
  /**
   * 手动中断AI回复
   */
  const interruptAIResponse = () => {
    try {
      if (conversationState.currentState === ConversationState.SPEAKING) {
        // 发送中断命令
        if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
          wsConnection.send(JSON.stringify({
            type: 'interrupt_response',
            data: {
              conversation_id: conversationState.conversationId,
              timestamp: Date.now()
            }
          }))
        }
        
        // 更新统计
        conversationState.stats.totalInterruptions++
        conversationState.interruptionCount++
        
        console.log('⚡ 手动中断AI回复')
      }
    } catch (error) {
      console.error('❌ 中断AI回复失败:', error)
    }
  }
  
  /**
   * 处理VAD结果（来自TEN框架）
   */
  const handleVADResult = (data) => {
    const { speech_detected, energy, timestamp } = data

    console.log('🎤 TEN VAD结果:', { speech_detected, energy })

    // 更新状态
    conversationState.speechDetected = speech_detected

    if (speech_detected) {
      conversationState.silenceDetected = false
      conversationState.lastUserSpeechTime = timestamp

      // 如果正在说话，检测中断
      if (conversationState.isSpeaking) {
        interruptAIResponse()
      }
    } else {
      conversationState.silenceDetected = true
    }
  }

  /**
   * 处理VAD事件
   */
  const handleVADEvent = (data) => {
    const { event_type, timestamp } = data
    
    if (event_type === 'speech_start') {
      conversationState.speechDetected = true
      conversationState.silenceDetected = false
      conversationState.lastUserSpeechTime = timestamp
      
      // 如果AI正在说话，检查是否需要中断
      if (conversationState.currentState === ConversationState.SPEAKING) {
        const timeSinceLastSpeech = timestamp - (conversationState.lastUserSpeechTime || 0)
        if (timeSinceLastSpeech <= conversationState.interruptionThreshold) {
          interruptAIResponse()
        }
      }
      
    } else if (event_type === 'speech_end') {
      conversationState.speechDetected = false
      conversationState.silenceDetected = true
    }
  }
  
  /**
   * 处理轮换检测事件
   */
  const handleTurnDetectionEvent = (data) => {
    const { turn_state, confidence, text } = data
    
    if (turn_state === 'finished' && confidence > 0.8) {
      // 用户说话结束，开始处理
      updateConversationState(ConversationState.PROCESSING)
      conversationState.finalTranscript = text
      conversationState.turnCount++
      
    } else if (turn_state === 'wait') {
      // 用户可能在等待
      updateConversationState(ConversationState.WAITING)
    }
  }
  
  /**
   * 处理ASR结果
   */
  const handleASRResult = (data) => {
    const { transcript, confidence, is_final } = data

    console.log('🎯 ASR结果:', { transcript, confidence, is_final })

    if (is_final) {
      conversationState.finalTranscript = transcript
      conversationState.realtimeTranscript = ''

      // 转换状态到处理中
      updateConversationState(ConversationState.PROCESSING)
    } else {
      conversationState.realtimeTranscript = transcript
    }
  }
  
  /**
   * 处理LLM响应
   */
  const handleLLMResponse = (data) => {
    const { response, conversation_id } = data

    console.log('🤖 LLM响应:', { response, conversation_id })

    // LLM响应完成，等待TTS事件来管理状态
    conversationState.lastAISpeechTime = Date.now()

    // 注意：状态转换现在由TTS事件处理
    // TTS开始时会转换到SPEAKING状态
    // TTS结束时会转换回LISTENING状态
  }
  
  /**
   * 处理TTS事件
   */
  const handleTTSEvent = (data) => {
    const { event_type, text, conversation_id } = data

    console.log('🔊 TTS事件:', { event_type, text })

    if (event_type === 'tts_start') {
      updateConversationState(ConversationState.SPEAKING)
      conversationState.isSpeaking = true
      console.log('🎵 开始TTS播放')

    } else if (event_type === 'tts_end') {
      conversationState.isSpeaking = false
      updateConversationState(ConversationState.LISTENING)
      console.log('🔄 TTS播放完成，回到监听状态')

    } else if (event_type === 'tts_interrupted') {
      conversationState.isSpeaking = false
      conversationState.stats.successfulInterruptions++
      updateConversationState(ConversationState.INTERRUPTED)
      console.log('⚡ TTS被中断')

      // 短暂延迟后转到监听状态
      setTimeout(() => {
        updateConversationState(ConversationState.LISTENING)
      }, 100)

    } else if (event_type === 'tts_error') {
      conversationState.isSpeaking = false
      updateConversationState(ConversationState.LISTENING)
      console.error('❌ TTS播放错误:', data.error)
    }
  }

  /**
   * 处理TTS音频数据
   */
  const handleTTSAudio = (data) => {
    const { audio, format, conversation_id } = data

    try {
      console.log('🎵 收到TTS音频数据')

      // 解码base64音频数据
      const audioBytes = atob(audio)
      const audioArray = new Uint8Array(audioBytes.length)
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i)
      }

      // 创建音频blob
      const audioBlob = new Blob([audioArray], { type: `audio/${format}` })
      const audioUrl = URL.createObjectURL(audioBlob)

      // 播放音频
      const audioElement = new Audio(audioUrl)
      audioElement.play().then(() => {
        console.log('🔊 TTS音频播放开始')
      }).catch(error => {
        console.error('❌ TTS音频播放失败:', error)
      })

      // 音频播放结束后清理URL
      audioElement.addEventListener('ended', () => {
        URL.revokeObjectURL(audioUrl)
        console.log('🔄 TTS音频播放结束')
      })

    } catch (error) {
      console.error('❌ TTS音频处理失败:', error)
    }
  }
  
  /**
   * 处理状态变化
   */
  const handleStateChange = (data) => {
    const { new_state, conversation_id, message } = data

    console.log('🔄 收到状态变化:', { new_state, conversation_id, message })

    // 只处理当前对话的状态变化
    if (conversation_id === conversationState.conversationId) {
      updateConversationState(new_state)

      // 如果TTS播放完成，确保恢复到监听状态
      if (new_state === 'listening' && message && message.includes('TTS播放完成')) {
        console.log('🔊 TTS播放完成，恢复监听状态')
        conversationState.isSpeaking = false
        conversationState.isListening = true
      }
    }
  }
  
  /**
   * 处理错误
   */
  const handleError = (data) => {
    const { message } = data
    conversationState.error = message
    console.error('❌ 服务器错误:', message)
    ElMessage.error(message)
  }
  
  /**
   * 更新对话状态
   */
  const updateConversationState = (newState) => {
    const oldState = conversationState.currentState
    conversationState.currentState = newState
    
    // 更新相关状态
    conversationState.isListening = newState === ConversationState.LISTENING
    conversationState.isProcessing = newState === ConversationState.PROCESSING
    conversationState.isSpeaking = newState === ConversationState.SPEAKING
    conversationState.isInterrupted = newState === ConversationState.INTERRUPTED
    
    console.log(`🔄 状态转换: ${oldState} -> ${newState}`)
  }
  
  /**
   * 获取状态描述
   */
  const getStateDescription = computed(() => {
    switch (conversationState.currentState) {
      case ConversationState.IDLE:
        return '点击开始语音对话'
      case ConversationState.LISTENING:
        return '正在监听，请说话...'
      case ConversationState.PROCESSING:
        return '正在处理您的话语...'
      case ConversationState.SPEAKING:
        return 'AI正在回复...'
      case ConversationState.INTERRUPTED:
        return '已中断，继续监听...'
      case ConversationState.WAITING:
        return '等待您继续说话...'
      default:
        return '未知状态'
    }
  })
  
  /**
   * 获取可视化状态
   */
  const getVisualizationState = computed(() => {
    if (conversationState.speechDetected) return 'recording'
    if (conversationState.isSpeaking) return 'speaking'
    if (conversationState.isListening) return 'listening'
    return 'idle'
  })
  
  /**
   * 清理资源
   */
  const cleanup = () => {
    // 停止对话
    stopVoiceConversation()

    // 停止心跳和连接检查
    stopHeartbeat()
    stopConnectionCheck()

    // 关闭WebSocket
    if (wsConnection) {
      wsConnection.close(1000, 'Client cleanup')
      wsConnection = null
    }

    // 清理音频资源
    if (vadProcessor) {
      vadProcessor.disconnect()
      vadProcessor = null
    }

    if (audioContext) {
      audioContext.close()
      audioContext = null
    }

    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop())
      audioStream = null
    }

    console.log('🧹 语音对话资源清理完成')
  }
  
  return {
    // 状态
    conversationState,
    
    // 计算属性
    getStateDescription,
    getVisualizationState,
    
    // 方法
    initVoiceConversation,
    startVoiceConversation,
    stopVoiceConversation,
    interruptAIResponse,
    cleanup
  }
}
