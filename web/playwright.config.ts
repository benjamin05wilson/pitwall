import { defineConfig } from '@playwright/test'
import path from 'node:path'
const python = process.env.PITWALL_PYTHON || path.resolve('../.venv/bin/python')
export default defineConfig({
  testDir: './tests', workers: 1, timeout: 30000,
  use: { baseURL: 'http://127.0.0.1:18743', viewport: { width: 1440, height: 1100 }, video: 'on' },
  webServer: {
    command: `"${python}" -m uvicorn pitwall.server:app --host 127.0.0.1 --port 18743`,
    url: 'http://127.0.0.1:18743/api/health', reuseExistingServer: false,
  },
})
