import React from "react";

const Guidelinepopup = ({ closePopup }) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
      <div
        className="bg-[#DFFFE2] p-6 rounded-lg shadow-xl w-72 text-center
      s:w-80
      md:w-1/2"
      >
        <h2 className="text-3xl font-semibold mb-4">
          How to use the Visualizer:
        </h2>
        <p className="text-lg font-normal mb-2">
          1. <span className="font-semibold">Upload File: </span>Click "Upload"
          to select an audio file.
        </p>
        <p className="text-lg font-normal mb-2">
          2. <span className="font-semibold">Analyze: </span>Hit "Analyze" to
          process the file and predict results.
        </p>
        <p className="text-lg font-normal mb-2">
          3. <span className="font-semibold">View Results: </span>Check the
          energy, valence, tension, and associated emotion.
        </p>
        <p className="text-lg font-normal mb-2">
          4. <span className="font-semibold">3D Plot: </span>Explore the 3D
          visualization.
        </p>
        <p className="text-lg font-normal mb-4">
          5. <span className="font-semibold">Reset: </span>Use "Reset" to upload
          and analyze a new file.
        </p>
        <button
          onClick={closePopup}
          className="px-6 py-2 bg-black text-white font-semibold rounded-lg hover:bg-red-600 transition duration-200"
        >
          Close
        </button>
      </div>
    </div>
  );
};

export default Guidelinepopup;
