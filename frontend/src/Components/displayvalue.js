import React from "react";

const DisplayValue = ({ label, value }) => {
  return (
    <div className="mb-3">
      <p className="text-base font-medium text-black mb-1">{label}</p>
      <div
        className="bg-white p-2 rounded-md border border-gray-200 mx-auto min-h-6 w-32 
      md:min-h-8 md:w-48"
      >
        {value !== undefined ? (
          <span className="text-base font-semibold">{value}</span>
        ) : (
          <span className="text-sm text-gray-400">Waiting...</span>
        )}
      </div>
    </div>
  );
};

export default DisplayValue;
