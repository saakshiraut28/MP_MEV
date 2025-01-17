import React, { useState } from "react";
import ReviewCard from "../Components/reviewcard";

const ReviewSection = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const reviewCards = [
    <ReviewCard
      key="1"
      reviewText="This is the best service I've ever used!"
      reviewerName="John Doe"
    />,
    <ReviewCard
      key="2"
      reviewText="Fantastic experience, highly recommend!"
      reviewerName="Jane Smith"
    />,
    <ReviewCard
      key="3"
      reviewText="Exceeded my expectations in every way."
      reviewerName="Michael Brown"
    />,
    <ReviewCard
      key="4"
      reviewText="Wonderful! Will definitely use again."
      reviewerName="Emily Davis"
    />,
    <ReviewCard
      key="5"
      reviewText="Absolutely loved it!"
      reviewerName="Chris Wilson"
    />,
    <ReviewCard
      key="6"
      reviewText="Incredible attention to detail."
      reviewerName="Sara Johnson"
    />,
  ];

  const handlePrev = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex === 0 ? reviewCards.length - 3 : prevIndex - 3
    );
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex + 3 >= reviewCards.length ? 0 : prevIndex + 3
    );
  };

  return (
    <div className="mx-auto p-6 bg-[#DFFFE2] h-auto" id="Reviews">
      <p className="font-bold text-2xl my-8">Happy Users!</p>
      <div className="flex items-center justify-between">
        <button
          className="bg-[#CBFFD0] hover:bg-[#CBFFD0] text-gray-700 font-bold py-2 px-4 rounded-full shadow-md border-2 border-green-200"
          onClick={handlePrev}
        >
          ←
        </button>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {reviewCards.slice(currentIndex, currentIndex + 3)}
        </div>
        <button
          className="bg-[#CBFFD0] hover:bg-[#CBFFD0] text-gray-700 font-bold py-2 px-4 rounded-full shadow-md border-2 border-green-200"
          onClick={handleNext}
        >
          →
        </button>
      </div>
    </div>
  );
};

export default ReviewSection;
