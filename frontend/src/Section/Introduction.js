import React, { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const Introduction = () => {
  const textRef = useRef(null);
  const paragraphRef = useRef(null);
  const introRef = useRef(null);

  useEffect(() => {
    const text = textRef.current;
    const chars = text.innerText.split("");
    text.innerHTML = chars
      .map((char) =>
        char === " "
          ? "<span class='inline-block'>&nbsp;</span>"
          : `<span class="inline-block">${char}</span>`
      )
      .join("");

    gsap.fromTo(
      text.querySelectorAll("span"),
      { opacity: 0, x: -10 },
      {
        opacity: 1,
        x: 0,
        duration: 0.1,
        stagger: 0.1,
        ease: "power2.out",
      }
    );

    gsap.fromTo(
      paragraphRef.current,
      { opacity: 0, y: 50 },
      {
        opacity: 1,
        y: 0,
        duration: 2,
        ease: "power3.out",
        delay: 2,
      }
    );

    gsap.to(introRef.current, {
      yPercent: -50,
      ease: "none",
      scrollTrigger: {
        trigger: introRef.current,
        start: "top top",
        end: "bottom top",
        scrub: true,
      },
    });
  }, []);

  return (
    <div
      ref={introRef}
      className="relative w-full h-screen overflow-hidden bg-[#85cef8]"
    >
      <div className="absolute top-0 left-0 w-[200%] h-full animate-smooth-wave blur-md">
        <svg
          className="w-full h-full"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 1200 200"
          preserveAspectRatio="none"
        >
          <path
            d="M0,100 C300,200 600,0 900,100 C1200,200 1500,0 1800,100 L1800,200 L0,200 Z"
            fill="#d7effd"
            opacity="0.7"
          />
        </svg>
      </div>

      <div className="relative flex items-center justify-center h-full z-10">
        <div className="overflow-hidden">
          <div>
            <div
              ref={textRef}
              className="text-center text-4xl font-semibold md:text-6xl"
            >
              Welcome to <span className="font-bold">EmoMeter</span>!
            </div>

            <div
              ref={paragraphRef}
              className="text-xl font-medium mt-4 ml-8 mr-8 whitespace-pre-wrap text-center break-keep break-words md:text-2xl md:px-4"
            >
              Your go-to tool for understanding the emotional heartbeat of
              music. Whether a song is energetic, tense, or calm, I’ve got it
              all covered. I take the audio file, extract intricate details of
              music, process them with precision, and deliver an understanding
              of a song’s mood. Join me on this journey to explore music like
              never before!
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Introduction;
