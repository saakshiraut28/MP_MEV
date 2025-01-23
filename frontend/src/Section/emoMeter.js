/** @format */

import React, { useState } from "react";
import Plot from "../Components/plot";
import DisplayValue from "../Components/displayvalue";
import { AlertCircle } from "lucide-react";

const EmoMeter = () => {
  const [fileName, setFileName] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      if (selectedFile.type.startsWith("audio/")) {
        setFileName(selectedFile.name);
        setFile(selectedFile);
        setError("");
      } else {
        setError("Only audio files are allowed!");
        event.target.value = "";
        setFileName("");
        setFile(null);
      }
    } else {
      setFileName("");
      setFile(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select an audio file first");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      setAnalysisResult(result);
    } catch (err) {
      setError(err.message || "Failed to analyze audio file");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFileName("");
    setFile(null);
    setError("");
    setAnalysisResult(null);
  };

  const formatPredictionValue = (value) => {
    if (value === undefined || value === null) return undefined;
    return typeof value === "number"
      ? `${(value * 100).toFixed(1)}%`
      : value.toString();
  };

  return (
    <div className="min-h-screen flex flex-col font-poppins text-black font-semibold pb-4">
      <div className="heading text-center py-8 space-y-2">
        <h1 className="text-3xl font-bold tracking-wide sm:text-4xl lg:text-5xl">
          EmoMeter
        </h1>
        <p className="text-base opacity-90 sm:text-lg lg:text-xl">
          A Music Emotion Visualizer
        </p>
      </div>

      {error && console.log(error)}

      <div className="flex flex-col gap-6 px-4 pb-12 md:flex-row md:items-start md:gap-8 md:px-8 lg:gap-12 lg:px-12">
        <div className="flex flex-col items-center space-y-4 p-4  rounded-xl backdrop-blur-sm md:w-1/3 md:sticky md:top-24">
          <div className="flex flex-col items-center gap-4 md:justify-center md:gap-6">
            <div className="relative flex items-center w-full ">
              <input
                type="file"
                id="audioUpload"
                accept="audio/*"
                className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
                onChange={handleFileChange}
                disabled={loading}
              />
              <label
                htmlFor="audioUpload"
                className={`w-56 rounded-lg font-poppins text-black
                          border-black border-2 p-2.5 text-sm font-semibold
                            text-center inline-block cursor-pointer
    ${loading ? "opacity-50 cursor-not-allowed" : "hover:bg-blue-100"}
                transition-all duration-200 md:text-base lg:text-lg`}
              >
                Upload Audio File
              </label>
            </div>

            <div className="flex flex-col items-center gap-4 w-full">
              {fileName && (
                <span className="ml-4 text-gray-700 text-sm md:text-base truncate w-full text-ellipsis text-center">
                  {fileName}
                </span>
              )}

              <button
                className={` w-56 rounded-lg font-poppins text-black hover:bg-blue-100 
                border-black border-2 p-2.5 text-sm font-semibold
                ${
                  loading ? "opacity-50 cursor-not-allowed" : "hover:opacity-90"
                }
                transition-opacity duration-200 md:text-base lg:text-lg`}
                onClick={handleAnalyze}
                disabled={loading}
              >
                {loading ? "Analyzing..." : "Analyze"}
              </button>

              <button
                className={` w-56 rounded-lg font-poppins text-black hover:bg-blue-100
                border-black border-2 p-2.5 text-sm font-semibold
                ${
                  loading ? "opacity-50 cursor-not-allowed" : "hover:opacity-90"
                }
                transition-opacity duration-200 md:text-base lg:text-lg`}
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            </div>
          </div>
        </div>

        <div className="md:w-2/3">
          <div className="mb-6">
            {analysisResult?.visualization_base64 && (
              <img
                src={`data:image/png;base64,${analysisResult.visualization_base64}`}
                alt="Emotion Analysis Visualization"
                className="w-full rounded-lg shadow-lg mb-4"
              />
            )}
            {analysisResult?.emotional_interpretation && (
              <p className="text-lg text-center px-4">
                {analysisResult.emotional_interpretation}
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-2 md:grid-cols-3">
            <DisplayValue
              label="Energy"
              value={formatPredictionValue(
                analysisResult?.predictions?.energy?.[0]
              )}
            />
            <DisplayValue
              label="Tension"
              value={formatPredictionValue(
                analysisResult?.predictions?.tension?.[0]
              )}
            />
            <DisplayValue
              label="Valence"
              value={formatPredictionValue(
                analysisResult?.predictions?.valence?.[0]
              )}
            />
            <DisplayValue
              label="Model Confidence"
              value={
                analysisResult?.r2_scores?.energy
                  ? `${(analysisResult.r2_scores.energy * 100).toFixed(1)}%`
                  : undefined
              }
            />
            <DisplayValue
              label="Analysis Type"
              value={analysisResult ? "Full Spectrum" : undefined}
            />
            <DisplayValue
              label="Processing Time"
              value={analysisResult ? "Real-time" : undefined}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmoMeter;
