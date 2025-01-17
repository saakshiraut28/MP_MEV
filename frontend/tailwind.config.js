/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      screens: {
        xs: "375px",
        s: "425px",
        md: "768px",
        lg: "1024px",
        xl: "1440px",
        "2xl": "1536px",
        // Custom breakpoints
        "3xl": "1920px",
        "4xl": "2560px",
      },
      colors: {
        customColor: "#f0f",
      },
      fontFamily: {
        poppins: ["Poppins", "sans-serif"], // Poppins font
      },
    },
  },
  plugins: [],
};
