import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
      imports: ['vue', '@vueuse/core'],
      dts: true
    }),
    Components({
      resolvers: [ElementPlusResolver()],
      dts: true
    })
  ],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:21007',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/tts': {
        target: 'http://localhost:21004',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/tts/, '')
      },
      '/asr': {
        target: 'http://localhost:21005',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/asr/, '')
      },
      '/s2s': {
        target: 'http://localhost:21006',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/s2s/, '')
      }
    }
  },
  css: {
    preprocessorOptions: {
      scss: {
        additionalData: `@import "./src/styles/variables.scss";`
      }
    }
  }
})
