import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the UI on :5173 and forwards /api to the FastAPI
// backend (python -m dashboard, default :8765). `npm run build` writes dist/,
// which the backend then serves itself.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: { manualChunks: { charts: ["recharts"] } },
    },
  },
  server: {
    proxy: { "/api": { target: process.env.DASHBOARD_API ?? "http://127.0.0.1:8765" } },
  },
});
