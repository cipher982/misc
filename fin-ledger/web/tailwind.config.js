/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f9ff',
          100: '#eaf2ff',
          200: '#cfe4ff',
          300: '#a6ccff',
          400: '#77adff',
          500: '#4a8bff',
          600: '#2b6cf0',
          700: '#1d51c4',
          800: '#1c4397',
          900: '#183a79'
        }
      }
    }
  },
  plugins: []
}
