import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg:      '#0d1117',
        surface: '#161b22',
        border:  '#30363d',
        text:    '#e6edf3',
        muted:   '#8b949e',
        low:     '#3fb950',
        medium:  '#f0883e',
        high:    '#ff7b72',
        accent:  '#58a6ff',
      },
      fontFamily: {
        sans: ['var(--font-outfit)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
      animation: {
        'pulse-red': 'pulse-red 2s ease-in-out infinite',
        'fade-in':   'fade-in 0.3s ease-out',
        shimmer:     'shimmer 1.5s infinite',
      },
      keyframes: {
        'pulse-red': {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(255, 123, 114, 0.4)' },
          '50%':       { boxShadow: '0 0 0 8px rgba(255, 123, 114, 0)' },
        },
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}

export default config
