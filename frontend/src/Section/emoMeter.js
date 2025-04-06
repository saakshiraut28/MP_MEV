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
  const [analysisType, setAnalysisType] = useState(null); // 'instrumental' or 'lyrics'

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
      setAnalysisType("instrumental");
    } catch (err) {
      setError(err.message || "Failed to analyze audio file");
    } finally {
      setLoading(false);
    }
  };

  const handleLyricAnalyze = async () => {
    if (!file) {
      setError("Please select an audio file first");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Optional: Add fast_mode parameter
      // You can add this as a state variable or UI toggle in your component
      formData.append("fast_mode", "true");

      const response = await fetch(
        "http://localhost:8000/analyze-lyrics-sentiment",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const result = await response.json();
      setAnalysisResult(result);
      setAnalysisType("lyrics");
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
    setAnalysisType(null);
  };

  const renderAnalysisResults = () => {
    if (!analysisResult) return null;

    if (analysisType === "instrumental") {
      // Render instrumental analysis results
      return (
        <>
          <div className="md:w-full flex flex-col justify-center items-center mb-6">
            {analysisResult.visualization_base64 && (
              <img
                src={`data:image/png;base64,${analysisResult.visualization_base64}`}
                alt="Emotion Analysis Visualization"
                className="w-1/2 rounded-lg shadow-lg mb-4"
              />
            )}
            {analysisResult.emotional_interpretation && (
              <p className="text-lg text-center px-4">
                {analysisResult.emotional_interpretation}
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-2 lg:grid-cols-3">
            <DisplayValue
              label="Energy"
              value={analysisResult?.predictions?.energy?.[0]?.toFixed(1)}
            />
            <DisplayValue
              label="Tension"
              value={analysisResult?.predictions?.tension?.[0]?.toFixed(1)}
            />
            <DisplayValue
              label="Valence"
              value={analysisResult?.predictions?.valence?.[0]?.toFixed(1)}
            />
            <DisplayValue
              label="Model Confidence"
              value={`${(analysisResult?.r2_scores?.energy * 100).toFixed(1)}%`}
            />
            <DisplayValue label="Analysis Type" value="Instrumental" />
            <DisplayValue label="Processing Time" value="Real-time" />
          </div>
        </>
      );
    } else {
      // Render lyrics-based analysis results
      const emotionEntries = Object.entries(analysisResult.emotions || {});
      const highScoringEmotions = emotionEntries.filter(
        ([emotion, score]) => score > 0.5
      );
      const highEmotionNames = highScoringEmotions.map(([emotion]) => emotion);

      return (
        <>
          <div className="md:w-full flex flex-col justify-center items-center mb-6">
            <div className="flex-1 gap-4 w-full justify-center">
              {analysisResult.waveform_base64 && (
                <img
                  src={`data:image/png;base64,${analysisResult.waveform_base64}`}
                  alt="Waveform Visualization"
                  className="w-full rounded-lg shadow-lg mb-4"
                />
              )}
              {analysisResult.visualization_base64 && (
                <img
                  src={`data:image/png;base64,${analysisResult.visualization_base64}`}
                  alt="Mel Spectrogram"
                  className="w-full rounded-lg shadow-lg mb-4"
                />
              )}
            </div>
            <h2 className="text-2xl font-bold mt-4 mb-2">
              Detected Emotion: {analysisResult.emotion}
            </h2>
          </div>

          <div className="grid grid-cols-2 gap-x-4 gap-y-2 lg:grid-cols-3">
            <DisplayValue
              label="Sentiment Score"
              value={analysisResult?.roberta_sentiment}
            />
            <DisplayValue
              label="Emotions Detected"
              value={highEmotionNames.join(", ")}
            />
            <DisplayValue label="Analysis Type" value="Lyrics-based" />
            <DisplayValue label="Processing Time" value="Real-time" />
          </div>
        </>
      );
    }
  };

  return (
    <div className="min-h-screen flex flex-col font-poppins text-black font-semibold pb-4 font-gloock">
      <div className="heading text-center py-8 space-y-2">
        <h1 className="text-3xl font-bold tracking-wide sm:text-4xl lg:text-5xl">
          EmoMeter
        </h1>
        <p className="text-base opacity-90 sm:text-lg lg:text-xl">
          A Music Emotion Visualizer
        </p>
      </div>

      {error && (
        <div className="flex items-center justify-center gap-2 text-red-500 mb-4">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      <div className="flex flex-col gap-6 px-4 pb-12 md:flex-row md:items-start md:gap-8 md:px-8 lg:gap-12 lg:px-12">
        <div className="flex flex-col items-center space-y-4 p-4 rounded-xl backdrop-blur-sm md:w-1/3 md:sticky md:top-24">
          <div className="flex flex-col items-center gap-4 w-full md:justify-center md:gap-6">
            <div className="flex flex-col items-center w-full max-w-[14rem]">
              <div className="relative flex items-center w-full">
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
                  className={`w-full rounded-lg font-poppins text-black
                            border-black border-2 p-2.5 text-sm font-semibold
                            text-center inline-block cursor-pointer
                            ${
                              loading
                                ? "opacity-50 cursor-not-allowed"
                                : "hover:bg-blue-100"
                            }
                            transition-all duration-200 md:text-base lg:text-lg`}
                >
                  Upload Audio File
                </label>
              </div>

              {fileName && (
                <span className="mt-4 text-gray-700 text-sm md:text-base truncate w-full text-center">
                  {fileName.length > 10
                    ? `${fileName.slice(0, 20)}...`
                    : fileName}
                </span>
              )}
            </div>

            <div className="flex flex-col items-center gap-4 w-full max-w-[14rem]">
              <button
                className={`w-full rounded-lg font-poppins text-black hover:bg-blue-100 
                          border-black border-2 p-2.5 text-sm font-semibold
                          ${
                            loading
                              ? "opacity-50 cursor-not-allowed"
                              : "hover:opacity-90"
                          }
                          transition-opacity duration-200 md:text-base lg:text-lg`}
                onClick={handleAnalyze}
                disabled={loading || !file}
              >
                {loading ? "Analyzing..." : "Analysis based on Instrumental"}
              </button>

              <button
                className={`w-full rounded-lg font-poppins text-black hover:bg-blue-100 
                          border-black border-2 p-2.5 text-sm font-semibold
                          ${
                            loading
                              ? "opacity-50 cursor-not-allowed"
                              : "hover:opacity-90"
                          }
                          transition-opacity duration-200 md:text-base lg:text-lg`}
                onClick={handleLyricAnalyze}
                disabled={loading || !file}
              >
                {loading ? "Analyzing..." : "Analysis Based on Lyrics"}
              </button>

              <button
                className={`w-full rounded-lg font-poppins text-black hover:bg-blue-100
                          border-black border-2 p-2.5 text-sm font-semibold
                          ${
                            loading
                              ? "opacity-50 cursor-not-allowed"
                              : "hover:opacity-90"
                          }
                          transition-opacity duration-200 md:text-base lg:text-lg`}
                onClick={handleReset}
                disabled={loading || (!file && !fileName && !analysisResult)}
              >
                Reset
              </button>
            </div>
          </div>
        </div>

        <div className="md:w-2/3">{renderAnalysisResults()}</div>
      </div>
    </div>
  );
};

export default EmoMeter;
