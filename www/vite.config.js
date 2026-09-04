import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: {
    exclude: ['@sauravpanda/flare'],
  },
  server: {
    fs: {
      allow: ['..'],
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
});

