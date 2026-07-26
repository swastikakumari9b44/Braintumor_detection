import { useState } from "react";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleFile = (file) => {
    setFile(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const uploadImage = async () => {
    if (!file) return alert("Upload image first");

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);

    try {
      const res = await axios.post("http://127.0.0.1:8000/predict", formData);
      setResult(res.data);
    } catch {
      alert("Backend error");
    }

    setLoading(false);
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.title}>🧠 Brain Tumor AI Analyzer</h1>

      {/* Upload Box */}
      <div
        style={{
          ...styles.dropZone,
          border: dragging ? "2px dashed #3b82f6" : "2px dashed #475569",
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        {preview ? (
          <img src={preview} style={styles.preview} />
        ) : (
          <p>Drag & Drop MRI Image Here</p>
        )}

        <input
          type="file"
          style={styles.hiddenInput}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      <button style={styles.button} onClick={uploadImage}>
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      {/* Result */}
      {result && (
        <div style={styles.result}>
          <h2>
            {result.prediction === "Tumor"
              ? "⚠️ Tumor Detected"
              : "✅ No Tumor"}
          </h2>

          <div style={styles.bar}>
            <div
              style={{
                ...styles.fill,
                width: `${result.confidence * 100}%`,
              }}
            />
          </div>

          <p>{(result.confidence * 100).toFixed(1)}% confidence</p>

          <div style={styles.images}>
            <div>
              <p>Original</p>
              <img src={preview} style={styles.img} />
            </div>

            <div>
              <p>AI Focus</p>
              <img
                src={`data:image/jpeg;base64,${result.heatmap}`}
                style={styles.img}
              />
            </div>
          </div>
        </div>
      )}

      {/* About */}
      <div style={styles.about}>
        <h3>About Model</h3>
        <p>
          CNN (MobileNetV2) trained on MRI scans. Grad-CAM highlights regions
          influencing prediction.
        </p>
        <p>Accuracy: ~90%</p>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #434965, #0d111c)",
    color: "white",
    textAlign: "center",
    padding: "40px",
    fontFamily: "system-ui",
  },

  title: {
    fontSize: "2.8rem",
    marginBottom: "30px",
  },

  dropZone: {
    width: "350px",
    height: "220px",
    margin: "auto",
    borderRadius: "15px",
    background: "rgba(255,255,255,0.05)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    position: "relative",
    backdropFilter: "blur(10px)",
  },

  hiddenInput: {
    position: "absolute",
    width: "100%",
    height: "100%",
    opacity: 0,
    cursor: "pointer",
  },

  preview: {
    width: "100%",
    height: "100%",
    objectFit: "cover",
    borderRadius: "15px",
  },

  button: {
    marginTop: "20px",
    padding: "12px 30px",
    borderRadius: "10px",
    background: "#3b82f6",
    border: "none",
    color: "white",
    fontSize: "16px",
    cursor: "pointer",
  },

  result: {
    marginTop: "30px",
    padding: "20px",
    background: "rgba(255,255,255,0.05)",
    width: "400px",
    marginInline: "auto",
    borderRadius: "15px",
  },

  bar: {
    height: "8px",
    background: "#1e293b",
    borderRadius: "10px",
    margin: "10px 0",
  },

  fill: {
    height: "8px",
    background: "#22c55e",
    borderRadius: "10px",
  },

  images: {
    display: "flex",
    justifyContent: "space-around",
    marginTop: "15px",
  },

  img: {
    width: "140px",
    borderRadius: "10px",
  },

  about: {
    marginTop: "40px",
    color: "#94a3b8",
  },
};

export default App;