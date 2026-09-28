import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  return {
    plugins: [react()],
    define: {
      'import.meta.env.VITE_API_BASE_URL': JSON.stringify(
        env.VITE_API_BASE_URL ?? (mode === 'development' ? '' : 'http://localhost:8000'),
      ),
    },
    server: {
      proxy: {
        '/api': 'http://localhost:8000',
      },
    },
  }
})
