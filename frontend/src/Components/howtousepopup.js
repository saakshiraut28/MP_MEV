import React, { useEffect } from "react";

const Guidelinepopup = ({ closePopup }) => {
  useEffect(() => {
    // Prevent body scrolling
    document.body.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = "unset";
    };
  }, []);

  return (
    <div
      className="fixed inset-0 z-[9999] flex items-center justify-center bg-black bg-opacity-50"
      onClick={closePopup}
    >
      <div
        className="bg-[#ADDEFB] p-6 rounded-lg shadow-2xl w-11/12 max-w-md text-center relative"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-2xl font-bold mb-4">How to Use EmoMeter</h2>
        {[
          "Upload audio file",
          "Click Analyze",
          "View emotional analysis",
          "Explore the plot",
          "Reset for new analysis",
        ].map((step, index) => (
          <p key={index} className="text-lg mb-2">
            <span className="font-semibold">{index + 1}. </span>
            {step}
          </p>
        ))}
        <button
          onClick={closePopup}
          className="mt-4 px-6 py-2 bg-[#1FA6F4] text-white rounded-lg hover:bg-[#0B94E3] transition"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default Guidelinepopup;
