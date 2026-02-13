import React, { useState, useRef } from "react";
import "./App.css";

function App() {
  const [transcript, setTranscript] = useState("");
  const [answer, setAnswer] = useState("");
  const [recording, setRecording] = useState(false);

  const [loadingTranscript, setLoadingTranscript] = useState(false);
  const [loadingAnswer, setLoadingAnswer] = useState(false); // kept for future use

  const [transcribeMs, setTranscribeMs] = useState(null);
  const [answerMs, setAnswerMs] = useState(null); // placeholder if you time /ask later

  const [mediaRecorder, setMediaRecorder] = useState(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        setTranscript("");
        setAnswer("");
        setTranscribeMs(null);
        setAnswerMs(null);

        setLoadingTranscript(true);

        const audioBlob = new Blob(chunksRef.current, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.wav");

       
        try {
          const res = await fetch("http://127.0.0.1:8000/transcribe_stream", {
            method: "POST",
            body: formData,
          });

          if (!res.body) {
            throw new Error("No stream body");
          }

          const reader = res.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = "";

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // NDJSON: split by line
            const lines = buffer.split("\n");
            // keep the last partial line in buffer
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) continue;
              try {
                const msg = JSON.parse(line);
                if (msg.type === "partial") {
                  setTranscript(msg.text || "");
                  setTranscribeMs(msg.elapsed_ms ?? null);
                } else if (msg.type === "final") {
                  setTranscript(msg.text || "");
                  setTranscribeMs(msg.elapsed_ms ?? null);
                }
              } catch (e) {
                // ignore bad lines
              }
            }
          }
        } catch (err) {
          setTranscript(" Streaming failed.");
        }

        setLoadingTranscript(false);

       
        if (transcript.trim().length > 0) {
          try {
            setLoadingAnswer(true);
            const t0 = performance.now();
            const askRes = await fetch("http://127.0.0.1:8000/ask", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ text: transcript }),
            });
            const askData = await askRes.json();
            setAnswer(askData.answer);
            setAnswerMs(Math.round(performance.now() - t0));
          } catch (error) {
            setAnswer(" Failed to generate answer.");
          }
          setLoadingAnswer(false);
        }
      };

      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
    } catch (err) {
      console.error("Recording error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      setRecording(false);
    }
  };

  return (
    <div className="container">
      <h1> RAG Voice Assistant</h1>

      <button onClick={startRecording} disabled={recording}>
         Start Recording
      </button>
      <button onClick={stopRecording} disabled={!recording}>
         Stop Recording
      </button>

      <div className="transcript-box">
        <p>
          <strong>Speech to Text:</strong>{" "}
          {loadingTranscript ? (
            <span className="loading-dots">
              Transcribing
            </span>
          ) : (
            transcript || "-"
          )}
          {" "}
          {transcribeMs !== null && (
            <span className="elapsed">({transcribeMs} ms)</span>
          )}
        </p>

        <p>
          <strong>Answer:</strong>{" "}
          {loadingAnswer ? (
            <span className="loading-dots">Generating answer</span>
          ) : (
            answer || "-"
          )}
          {" "}
          {answerMs !== null && (
            <span className="elapsed">({answerMs} ms)</span>
          )}
        </p>
      </div>
    </div>
  );
}

export default App;

