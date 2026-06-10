import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        variant: {
          bm25: "#6366f1",
          dense: "#10b981",
          hybrid: "#f59e0b",
          rerank: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
export default config;
