import { ref, reactive } from 'vue'

// 全局语音状态
const voiceState = reactive({
  isRecording: false,
  isListening: false,
  isSpeaking: false,
  isProcessing: false,
  isMuted: false,
  volume: 0,
  transcript: '',
  error: null
})

// 媒体相关引用
let mediaRecorder = null
let audioContext = null
let analyser = null
let microphone = null
let audioChunks = []
let speechSynthesis = null
let currentUtterance = null

export function useVoice() {
  /**
   * 初始化语音功能
   */
  const initVoice = async () => {
    try {
      // 检查浏览器支持
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('浏览器不支持录音功能')
      }

      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      })

      // 初始化音频上下文
      audioContext = new (window.AudioContext || window.webkitAudioContext)()
      analyser = audioContext.createAnalyser()
      microphone = audioContext.createMediaStreamSource(stream)
      
      analyser.fftSize = 256
      microphone.connect(analyser)

      // 初始化媒体录制器，使用更兼容的格式
      let mimeType = 'audio/webm;codecs=opus'
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm'
      }
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/mp4'
      }

      mediaRecorder = new MediaRecorder(stream, {
        mimeType: mimeType,
        audioBitsPerSecond: 128000  // 设置音频比特率
      })

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' })
        audioChunks = []
        processAudioBlob(audioBlob)
      }

      // 初始化语音合成
      speechSynthesis = window.speechSynthesis

      console.log('语音功能初始化成功')
      return true
    } catch (error) {
      console.error('语音功能初始化失败:', error)
      voiceState.error = error.message
      return false
    }
  }

  /**
   * 开始录音
   */
  const startRecording = async () => {
    try {
      if (!mediaRecorder) {
        await initVoice()
      }

      if (mediaRecorder && mediaRecorder.state === 'inactive') {
        voiceState.isRecording = true
        voiceState.isListening = true
        voiceState.transcript = ''
        voiceState.error = null
        
        audioChunks = []
        mediaRecorder.start(1000) // 每1秒收集一次数据，减少数据碎片
        
        // 开始音量监测
        startVolumeMonitoring()
        
        console.log('开始录音')
      }
    } catch (error) {
      console.error('开始录音失败:', error)
      voiceState.error = error.message
      voiceState.isRecording = false
      voiceState.isListening = false
    }
  }

  /**
   * 停止录音
   */
  const stopRecording = () => {
    try {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        voiceState.isRecording = false
        voiceState.isListening = false
        voiceState.isProcessing = true
        
        mediaRecorder.stop()
        stopVolumeMonitoring()
        
        console.log('停止录音')
      }
    } catch (error) {
      console.error('停止录音失败:', error)
      voiceState.error = error.message
    }
  }

  /**
   * 处理音频数据
   */
  const processAudioBlob = async (audioBlob) => {
    try {
      // 创建FormData发送到后端
      const formData = new FormData()
      formData.append('file', audioBlob, 'recording.webm')

      const response = await fetch('http://localhost:8000/api/speech-to-text', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`语音识别失败: ${response.status}`)
      }

      const result = await response.json()
      voiceState.transcript = result.text || ''

      console.log('语音识别结果:', voiceState.transcript)

      // 如果有识别结果，发送到聊天API
      if (voiceState.transcript.trim()) {
        await sendVoiceToChat(voiceState.transcript)
      }

    } catch (error) {
      console.error('处理音频失败:', error)
      voiceState.error = error.message
    } finally {
      voiceState.isProcessing = false
      voiceState.isListening = false  // 确保重置监听状态
    }
  }

  /**
   * 发送语音转录到聊天
   */
  const sendVoiceToChat = async (text) => {
    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: text }],
          stream: false,
          temperature: 0.7
        })
      })

      if (!response.ok) {
        throw new Error(`聊天请求失败: ${response.status}`)
      }

      const result = await response.json()
      const responseText = result.response || '抱歉，我无法理解您的问题。'
      
      // 将回复转换为语音
      await textToSpeech(responseText)
      
    } catch (error) {
      console.error('发送聊天失败:', error)
      voiceState.error = error.message
    }
  }

  /**
   * 文本转语音
   */
  const textToSpeech = async (text) => {
    try {
      voiceState.isSpeaking = true
      
      // 方法1: 使用后端TTS服务
      const response = await fetch('http://localhost:8000/api/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text,
          voice: 'default',
          speed: 1.0
        })
      })

      if (response.ok) {
        const contentType = response.headers.get('content-type')

        if (contentType && contentType.includes('application/json')) {
          const result = await response.json()

          // 检查是否是特殊的浏览器TTS响应
          if (result.use_browser_tts) {
            console.log('使用增强的浏览器TTS:', result.text)
            await browserTextToSpeech(result.text, result.lang || 'zh-CN')
            return
          }

          // 检查是否有音频URL
          if (result.audio_url) {
            await playAudio(result.audio_url)
            return
          }
        } else {
          // 可能是音频数据，尝试播放
          const audioBlob = await response.blob()
          if (audioBlob.size > 0) {
            const audioUrl = URL.createObjectURL(audioBlob)
            await playAudio(audioUrl)
            return
          }
        }
      }

      // 方法2: 使用浏览器内置TTS作为备选
      await browserTextToSpeech(text)
      
    } catch (error) {
      console.error('语音合成失败:', error)
      // 降级到浏览器TTS
      await browserTextToSpeech(text)
    }
  }

  /**
   * 浏览器内置TTS
   */
  const browserTextToSpeech = async (text, lang = 'zh-CN') => {
    return new Promise((resolve, reject) => {
      if (!speechSynthesis) {
        reject(new Error('浏览器不支持语音合成'))
        return
      }

      // 停止当前播放
      speechSynthesis.cancel()

      currentUtterance = new SpeechSynthesisUtterance(text)
      currentUtterance.lang = lang
      currentUtterance.rate = 1.0
      currentUtterance.pitch = 1.0
      currentUtterance.volume = voiceState.isMuted ? 0 : 1

      currentUtterance.onstart = () => {
        voiceState.isSpeaking = true
      }

      currentUtterance.onend = () => {
        voiceState.isSpeaking = false
        resolve()
      }

      currentUtterance.onerror = (error) => {
        voiceState.isSpeaking = false
        reject(error)
      }

      speechSynthesis.speak(currentUtterance)
    })
  }

  /**
   * 播放音频文件
   */
  const playAudio = async (audioUrl) => {
    return new Promise((resolve, reject) => {
      const audio = new Audio(audioUrl)
      
      audio.onloadstart = () => {
        voiceState.isSpeaking = true
      }
      
      audio.onended = () => {
        voiceState.isSpeaking = false
        resolve()
      }
      
      audio.onerror = (error) => {
        voiceState.isSpeaking = false
        reject(error)
      }
      
      audio.volume = voiceState.isMuted ? 0 : 1
      audio.play()
    })
  }

  /**
   * 停止语音播放
   */
  const stopSpeaking = () => {
    voiceState.isSpeaking = false
    
    // 停止浏览器TTS
    if (speechSynthesis) {
      speechSynthesis.cancel()
    }
    
    // 停止所有音频元素
    const audioElements = document.querySelectorAll('audio')
    audioElements.forEach(audio => {
      audio.pause()
      audio.currentTime = 0
    })
  }

  /**
   * 切换静音状态
   */
  const toggleMute = () => {
    voiceState.isMuted = !voiceState.isMuted
    
    // 如果正在播放，调整音量
    if (currentUtterance) {
      currentUtterance.volume = voiceState.isMuted ? 0 : 1
    }
  }

  /**
   * 开始音量监测
   */
  const startVolumeMonitoring = () => {
    if (!analyser) return

    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const updateVolume = () => {
      if (!voiceState.isRecording) return

      analyser.getByteFrequencyData(dataArray)
      
      let sum = 0
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i]
      }
      
      voiceState.volume = sum / bufferLength / 255
      
      requestAnimationFrame(updateVolume)
    }

    updateVolume()
  }

  /**
   * 停止音量监测
   */
  const stopVolumeMonitoring = () => {
    voiceState.volume = 0
  }

  /**
   * 清理资源
   */
  const cleanup = () => {
    stopRecording()
    stopSpeaking()
    
    if (audioContext) {
      audioContext.close()
      audioContext = null
    }
    
    if (mediaRecorder) {
      mediaRecorder = null
    }
  }

  return {
    voiceState,
    initVoice,
    startRecording,
    stopRecording,
    textToSpeech,
    stopSpeaking,
    toggleMute,
    cleanup
  }
}
