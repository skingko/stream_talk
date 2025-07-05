import { ref } from 'vue'
import axios from 'axios'

// 配置axios
const api = axios.create({
  baseURL: 'http://localhost:8002',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 全局状态
const isGenerating = ref(false)
const currentStream = ref(null)

export function useChat() {
  /**
   * 发送聊天消息
   * @param {Object} message - 消息对象
   * @param {string} mode - 模式 ('text' | 'voice')
   * @returns {Promise<Object>} 响应对象
   */
  const sendMessage = async (message, mode = 'text') => {
    try {
      isGenerating.value = true
      
      if (mode === 'text') {
        return await sendTextMessage(message)
      } else {
        return await sendVoiceMessage(message)
      }
    } catch (error) {
      console.error('发送消息失败:', error)
      throw error
    } finally {
      isGenerating.value = false
    }
  }

  /**
   * 发送文本消息
   */
  const sendTextMessage = async (message) => {
    const requestData = {
      messages: [
        {
          role: 'user',
          content: message.content
        }
      ],
      stream: false,  // 改为非流式，简化处理
      temperature: 0.7,
      max_new_tokens: 1024  // 修正参数名
    }

    // 如果有文件，处理多模态请求
    if (message.files && message.files.length > 0) {
      return await sendMultimodalMessage(message)
    }

    console.log('发送聊天请求:', requestData)
    const response = await api.post('/api/chat', requestData)
    console.log('收到聊天响应:', response.data)

    return {
      content: response.data.response || '抱歉，我无法理解您的问题。',
      type: 'text'
    }
  }

  /**
   * 发送语音消息
   */
  const sendVoiceMessage = async (message) => {
    // 语音消息处理逻辑
    const response = await api.post('/api/voice/process', message.audioData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    const transcript = response.data.transcript
    
    // 将转录的文本发送给聊天API
    const chatResponse = await sendTextMessage({
      content: transcript,
      type: 'text'
    })

    // 将回复转换为语音
    const ttsResponse = await api.post('/api/text-to-speech', {
      text: chatResponse.content,
      voice: 'default',
      speed: 1.0
    })

    return {
      content: chatResponse.content,
      type: 'voice',
      audioUrl: ttsResponse.data.audio_url,
      transcript: transcript
    }
  }

  /**
   * 发送多模态消息
   */
  const sendMultimodalMessage = async (message) => {
    const formData = new FormData()
    
    // 添加文本内容
    if (message.content) {
      formData.append('text', message.content)
    }
    
    // 添加文件
    message.files.forEach((fileInfo, index) => {
      if (fileInfo.type === 'image') {
        formData.append('image', fileInfo.file)
      } else if (fileInfo.type === 'video') {
        formData.append('video', fileInfo.file)
      } else if (fileInfo.type === 'audio') {
        formData.append('audio', fileInfo.file)
      } else {
        formData.append('file', fileInfo.file)
      }
    })
    
    formData.append('temperature', '0.7')

    // 根据文件类型选择合适的API端点
    const hasImage = message.files.some(f => f.type === 'image')
    const endpoint = hasImage ? '/api/image-text' : '/api/multimodal'

    const response = await api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    return {
      content: response.data.result || response.data.message || '处理完成',
      type: 'text'
    }
  }

  /**
   * 流式聊天
   */
  const sendStreamMessage = async (message, onChunk) => {
    try {
      isGenerating.value = true
      
      const requestData = {
        messages: [
          {
            role: 'user',
            content: message.content
          }
        ],
        stream: true,
        temperature: 0.7,
        max_tokens: 1024
      }

      const response = await fetch('http://localhost:8002/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestData)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      currentStream.value = reader

      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // 保留最后一个可能不完整的行
        
        for (const line of lines) {
          if (line.trim()) {
            try {
              const data = JSON.parse(line)
              if (data.content) {
                onChunk(data.content)
              }
            } catch (e) {
              // 忽略解析错误，继续处理下一行
              onChunk(line)
            }
          }
        }
      }
    } catch (error) {
      console.error('流式聊天失败:', error)
      throw error
    } finally {
      isGenerating.value = false
      currentStream.value = null
    }
  }

  /**
   * 停止生成
   */
  const stopGeneration = () => {
    if (currentStream.value) {
      currentStream.value.cancel()
      currentStream.value = null
    }
    isGenerating.value = false
  }

  /**
   * 上传文件
   */
  const uploadFile = async (file) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })

    return response.data
  }

  /**
   * 健康检查
   */
  const checkHealth = async () => {
    try {
      const response = await api.get('/health')
      return response.data
    } catch (error) {
      console.error('健康检查失败:', error)
      return null
    }
  }

  return {
    sendMessage,
    sendStreamMessage,
    stopGeneration,
    uploadFile,
    checkHealth,
    isGenerating: isGenerating.value
  }
}
