import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// 개발 서버 포트: VM 사용 규칙에서 화면 담당(박단비)에게 배정된 8504번
export default defineConfig({
  plugins: [react()],
  server: {
    port: 8504,
    strictPort: true,
  },
})
