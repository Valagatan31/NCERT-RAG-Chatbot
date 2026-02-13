import React from "react";
import "./Loader.css";

const Loader = () => {
  return (
    <div className="loader-overlay">
      <div className="loader"></div>
      <p>Processing... Please wait</p>
    </div>
  );
};

export default Loader;
