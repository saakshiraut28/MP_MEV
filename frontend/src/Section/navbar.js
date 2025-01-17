import React, { useState } from "react";
import MyComponent from "../Components/howtousepopup";

const Navbar = () => {
  const [showPopup, setShowPopup] = useState(false);

  const handlePopupToggle = (e) => {
    e.preventDefault();
    setShowPopup(!showPopup);
  };

  return (
    <nav className="navbar bg-[#FADADD] sticky top-0 z-50 shadow-sm">
      <div
        className="max-w-7xl mx-auto px-4 flex justify-center items-center font-poppins text-black text-xs font-semibold
        xs:text-sm
        md:text-lg
        lg:text-xl"
      >
        <a
          className="m-4 hover:opacity-70 transition-opacity duration-200
          md:mx-8
          lg:mx-12"
          href="#EmoMeter"
        >
          EmoMeter
        </a>
        <a
          className="m-4 hover:opacity-70 transition-opacity duration-200
          md:mx-8
          lg:mx-12"
          href="#Guidelines"
          onClick={handlePopupToggle}
        >
          How to use?
        </a>
        <a
          className="m-4 hover:opacity-70 transition-opacity duration-200
          md:mx-8
          lg:mx-12"
          href="#Reviews"
        >
          Reviews
        </a>
        <a
          className="m-4 hover:opacity-70 transition-opacity duration-200
          md:mx-8
          lg:mx-12"
          href="#Aboutus"
        >
          About us
        </a>
      </div>

      {showPopup && <MyComponent closePopup={() => setShowPopup(false)} />}
    </nav>
  );
};

export default Navbar;
