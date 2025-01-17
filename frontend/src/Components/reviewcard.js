import React from "react";

const ReviewCard = ({ reviewText, reviewerName }) => {
  return (
    <div className="max-w-sm mx-auto p-6 bg-white rounded-lg text-center border-black border-[1px]">
      <p className="text-gray-700 text-base mb-4">{reviewText}</p>
      <p className="text-gray-900 font-semibold">- {reviewerName}</p>
    </div>
  );
};

export default ReviewCard;
