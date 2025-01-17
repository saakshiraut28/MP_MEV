import "./App.css";
import Navbar from "./Section/navbar";
import EmoMeter from "./Section/emoMeter";
import Aboutus from "./Section/Aboutsection";
import ReviewSection from "./Section/reviewSection";
import Footer from "./Section/footer";

function App() {
  return (
    <div className="App min-h-screen">
      <Navbar />
      <EmoMeter />
      <ReviewSection />
      <Aboutus />
      <Footer />
    </div>
  );
}

export default App;
