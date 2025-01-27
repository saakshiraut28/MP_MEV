import "./App.css";
import Navbar from "./Section/navbar";
import EmoMeter from "./Section/emoMeter";
import Aboutus from "./Section/Aboutsection";
import ReviewSection from "./Section/reviewSection";
import Footer from "./Section/footer";
import Introduction from "./Section/Introduction";

function App() {
  return (
    <div className="App relative w-full min-h-screen overflow-x-hidden">
      <div className="absolute top-0 left-0 w-[200%] h-full animate-smooth-wave blur-md -z-10 bg-white">
        <svg
          className="w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1200 200"
          preserveAspectRatio="none"
        >
          <path
            d="M0,100 C300,200 600,0 900,100 C1200,200 1500,0 1800,100 L1800,200 L0,200 Z"
            fill="#6BC4F7"
            opacity="0.7"
          />
        </svg>
      </div>

      <div className="relative">
        <Introduction />
      </div>

      <div className="relative z-10">
        <Navbar />
        <EmoMeter />
        <ReviewSection />
        <Aboutus />
        <Footer />
      </div>
    </div>
  );
}

export default App;
