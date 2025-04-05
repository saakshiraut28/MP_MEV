import React, { useState } from "react";
import ReviewCard from "../Components/reviewcard";

const ReviewSection = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  const reviewCards = [
    <ReviewCard
      key="1"
      reviewText="Really loved this project. Lol, I tested a few song randomly. Please do this for Hindi songs"
      reviewerName="Shruti Pote"
    />,
    <ReviewCard
      key="2"
      reviewText="This project idea is very unique. Being a AI student, I never saw a project things so much about Music. It's something we listen to everyday. But I never thaought of this aspects of it."
      reviewerName="Samiksha Patil"
    />,
    <ReviewCard
      key="3"
      reviewText="Hi, I'm CS student. Saakshi shared this with me to review it. And really think this is good idea. I would love to contribute to it to enhance its feature."
      reviewerName="Sumedh Pardeshi"
    />,
    <ReviewCard
      key="4"
      reviewText="I like this project. I'm planning to implement something like this next year for my major project. thanks shrimayee for sharing."
      reviewerName="Nikhil Sharma"
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
    <div className="mx-auto p-6 h-auto font-gloock" id="Reviews">
      <p className="font-bold text-2xl my-8">Happy Users!</p>
      <div className="flex items-center justify-between">
        <button
          className=" hover:bg-[#d7effd] text-black font-bold text-lg py-2 px-4 rounded-full shadow-md border-2 border-black"
          onClick={handlePrev}
        >
          ←
        </button>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {reviewCards.slice(currentIndex, currentIndex + 3)}
        </div>
        <button
          className=" hover:bg-[#d7effd] text-black font-bold text-lg py-2 px-4 rounded-full shadow-md border-2 border-black"
          onClick={handleNext}
        >
          →
        </button>
      </div>
    </div>
  );
};

export default ReviewSection;
