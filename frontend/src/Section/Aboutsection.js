import React from "react";
import Saakshi from "../Assets/Saakshi_Raut.png";
import Shrimayee from "../Assets/Shrimayee_Mishra.png";
import Arfia from "../Assets/Arfia_Shaikh.png";
import TeamCard from "../Components/teamCard";

const Aboutus = () => {
  return (
    <section
      className="relative flex flex-col font-poppin text-black px-4 py-8 items-center font-gloock
      lg:px-24 lg:py-8 lg:flex-row"
      id="Aboutus"
    >
      <div className="absolute top-0 left-0 w-full h-full -z-10 overflow-hidden">
        <svg
          className="w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1200 200"
          preserveAspectRatio="none"
        >
          <path
            d="M0,100 C300,200 600,0 900,100 C1200,200 1500,0 1800,100 L1800,200 L0,200 Z"
            fill="#fff"
            opacity="1"
          />
        </svg>
      </div>

      <div
        className="flex flex-col ml-4
      lg:w-[49%]"
      >
        <div
          className="about-para pl-2 text-black
        lg:ml-4"
        >
          <p
            className="pt-6 pb-4 text-xl font-bold py-4
          md:text-3xl"
          >
            About Us
          </p>
          <p
            className="mr-4 text-lg text-justify font-medium 
          lg:text-xl"
          >
            EmoMeter is an innovative tool that analyzes the emotional aspects
            of music. By simply uploading an audio file, users can instantly see
            predictions for energy, valence, and tension of the song, along with
            its corresponding emotion. The system uses a trained model to
            process the audio features and provides a 3D visualization to help
            users explore the emotional landscape of their favorite tracks.
          </p>
          <p
            className="mr-4 text-lg text-justify font-medium
          lg:text-xl"
          >
            EmoMeter is a system developed by a team of final year computer
            engineering student as their final year project.
          </p>
        </div>
      </div>

      <div
        className="pt-8 pb-8 flex justify-center items-center
      lg:w-1/2"
      >
        <div
          className="p-4 pt-8 flex flex-col space-y-24
          md:flex md:flex-row md:space-y-0 md:p-0 md:space-x-8
      lg:flex-col lg:space-y-20"
        >
          <TeamCard
            fullName="Shrimayee Mishra"
            role="Computer Engineering Student"
            imgLink={Shrimayee}
          />
          <TeamCard
            fullName="Arfia Shaikh"
            role="Computer Engineering Student"
            imgLink={Arfia}
          />
        </div>
        <div
          className="p-4 flex flex-col items-center"
          style={{ marginLeft: "10px" }}
        >
          <TeamCard
            fullName="Saakshi Raut"
            role="Computer Engineering Student"
            imgLink={Saakshi}
          />
        </div>
      </div>
    </section>
  );
};

export default Aboutus;
