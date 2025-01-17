import React from "react";

const Plot = () => {
  return (
    <div
      className="h-60 w-11/12 bg-white border-black border-2 rounded-lg p-4 mx-auto my-4
        shadow-lg transition-all duration-300
        sm:w-4/5
        md:h-72 md:my-6
        lg:h-80
        xl:h-96 xl:w-3/4"
    >
      <div className="h-full w-full flex flex-col items-center justify-center">
        <svg
          className="w-16 h-16 mb-4"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <g id="SVGRepo_bgCarrier" stroke-width="0"></g>
          <g
            id="SVGRepo_tracerCarrier"
            stroke-linecap="round"
            stroke-linejoin="round"
          ></g>
          <g id="SVGRepo_iconCarrier">
            {" "}
            <path
              d="M12 19.5C12 20.8807 10.8807 22 9.5 22C8.11929 22 7 20.8807 7 19.5C7 18.1193 8.11929 17 9.5 17C10.8807 17 12 18.1193 12 19.5Z"
              stroke="#1C274C"
              stroke-width="1.5"
            ></path>{" "}
            <path
              d="M22 17.5C22 18.8807 20.8807 20 19.5 20C18.1193 20 17 18.8807 17 17.5C17 16.1193 18.1193 15 19.5 15C20.8807 15 22 16.1193 22 17.5Z"
              stroke="#1C274C"
              stroke-width="1.5"
            ></path>{" "}
            <path
              d="M22 8L12 12"
              stroke="#1C274C"
              stroke-width="1.5"
              stroke-linecap="round"
            ></path>{" "}
            <path
              d="M14.4556 5.15803L14.7452 5.84987L14.4556 5.15803ZM16.4556 4.32094L16.1661 3.62909L16.4556 4.32094ZM21.1081 3.34059L20.6925 3.96496L20.6925 3.96496L21.1081 3.34059ZM12.75 19.0004V8.84787H11.25V19.0004H12.75ZM22.75 17.1542V8.01078H21.25V17.1542H22.75ZM14.7452 5.84987L16.7452 5.01278L16.1661 3.62909L14.1661 4.46618L14.7452 5.84987ZM22.75 8.01078C22.75 6.67666 22.752 5.59091 22.6304 4.76937C22.5067 3.93328 22.2308 3.18689 21.5236 2.71622L20.6925 3.96496C20.8772 4.08787 21.0473 4.31771 21.1466 4.98889C21.248 5.67462 21.25 6.62717 21.25 8.01078H22.75ZM16.7452 5.01278C18.0215 4.47858 18.901 4.11263 19.5727 3.94145C20.2302 3.77391 20.5079 3.84204 20.6925 3.96496L21.5236 2.71622C20.8164 2.24554 20.0213 2.2792 19.2023 2.48791C18.3975 2.69298 17.3967 3.114 16.1661 3.62909L16.7452 5.01278ZM12.75 8.84787C12.75 8.18634 12.751 7.74991 12.7875 7.41416C12.822 7.09662 12.8823 6.94006 12.9594 6.8243L11.7106 5.99325C11.4527 6.38089 11.3455 6.79864 11.2963 7.25218C11.249 7.68752 11.25 8.21893 11.25 8.84787H12.75ZM14.1661 4.46618C13.5859 4.70901 13.0953 4.91324 12.712 5.12494C12.3126 5.34549 11.9686 5.60562 11.7106 5.99325L12.9594 6.8243C13.0364 6.70855 13.1575 6.59242 13.4371 6.438C13.7328 6.27473 14.135 6.10528 14.7452 5.84987L14.1661 4.46618Z"
              fill="#1C274C"
            ></path>{" "}
            <path
              d="M7 11V6.5V2"
              stroke="#1C274C"
              stroke-width="1.5"
              stroke-linecap="round"
            ></path>{" "}
            <circle
              cx="4.5"
              cy="10.5"
              r="2.5"
              stroke="#1C274C"
              stroke-width="1.5"
            ></circle>{" "}
            <path
              d="M10 5C8.75736 5 7 4.07107 7 2"
              stroke="#1C274C"
              stroke-width="1.5"
              stroke-linecap="round"
            ></path>{" "}
          </g>
        </svg>
        <p className="text-center font-medium mb-2 text-gray-500">
          Upload your audio file
        </p>
        <p className="text-center text-sm opacity-75 max-w-md text-gray-500">
          Your emotion visualization will appear here after analysis
        </p>
      </div>
    </div>
  );
};

export default Plot;
