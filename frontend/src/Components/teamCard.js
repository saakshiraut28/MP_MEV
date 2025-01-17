import React from "react";

const TeamCard = ({ fullName, role, imgLink }) => {
  return (
    <div className="flex flex-col items-center space-y-2 text-base font-medium lg:text-xl">
      <div className="w-32 h-32 bg-blue-500 rounded-full">
        <img
          className="w-32 h-32 object-cover rounded-full"
          src={imgLink}
          alt={fullName}
        />
      </div>
      <p className="text-center text-pretty font-bold">{fullName}</p>
      <p className="text-center">{role}</p>
    </div>
  );
};

export default TeamCard;
