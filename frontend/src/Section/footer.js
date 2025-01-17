import React from "react";

const Footer = () => {
  return (
    <footer className="bg-[#DFFFE2] py-6 text-center">
      <div className="max-w-4xl mx-auto">
        <p className="text-gray-700 text-lg mb-4">
          Have questions or feedback? Reach out to us at:
          <a
            href="mailto:shrimayeemishra934@gmail.com"
            className="text-blue-600 underline ml-1"
          >
            shrimayeemishra934@gmail.com
          </a>
        </p>
        <p className="text-gray-600 text-sm">
          © 2025 EmoMeter. All rights reserved. <br /> Powered by SAS
        </p>
      </div>
    </footer>
  );
};

export default Footer;
