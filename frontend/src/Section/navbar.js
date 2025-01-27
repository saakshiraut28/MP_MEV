import React, { useState, useEffect } from "react";
import MyComponent from "../Components/howtousepopup";

const Navbar = () => {
  const [showPopup, setShowPopup] = useState(false);
  const [isNavbarVisible, setIsNavbarVisible] = useState(false);

  const handlePopupToggle = (e) => {
    e.preventDefault();
    setShowPopup(true);
  };

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > window.innerHeight * 0.4) {
        setIsNavbarVisible(true);
      } else {
        setIsNavbarVisible(false);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  return (
    <>
      <nav
        className={`navbar fixed top-0 left-0 w-full z-50 transition-transform duration-300 ${
          isNavbarVisible ? "translate-y-0" : "-translate-y-full"
        }`}
      >
        <div className="bg-[#fff] bg-opacity-70 backdrop-blur-md shadow-md py-1">
          <div
            className="max-w-7xl mx-auto px-4 flex justify-center items-center font-poppins text-black text-xs font-semibold
          xs:text-sm
          md:text-lg
          lg:text-xl"
          >
            <a
              className="m-4 hover:opacity-70 transition-opacity"
              href="#EmoMeter"
            >
              EmoMeter
            </a>
            <a
              className="m-4 hover:opacity-70 transition-opacity"
              href="#Guidelines"
              onClick={handlePopupToggle}
            >
              How to use?
            </a>
            <a
              className="m-4 hover:opacity-70 transition-opacity"
              href="#Reviews"
            >
              Reviews
            </a>
            <a
              className="m-4 hover:opacity-70 transition-opacity"
              href="#Aboutus"
            >
              About us
            </a>
          </div>
        </div>
      </nav>

      {showPopup && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center">
          <MyComponent closePopup={() => setShowPopup(false)} />
        </div>
      )}
    </>
  );
};

export default Navbar;
