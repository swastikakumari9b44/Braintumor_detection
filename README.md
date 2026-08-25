# 🧠 Brain Tumor Detection

### Explainable AI for Brain MRI Classification

An end-to-end computer vision application that analyzes **brain MRI scans** and predicts whether an image is classified as **Tumor** or **No Tumor**.

The application combines a **TensorFlow/Keras deep-learning model** with **Grad-CAM explainability**, allowing users to see not only the model's prediction and confidence score, but also the regions of the MRI that contributed to the prediction.

Built with **React + Vite** on the frontend and **FastAPI + TensorFlow/Keras** on the backend.

> **Upload an MRI → Get a prediction → Understand what the model focused on**

---

# ✨ Key Features

### 🔍 Binary MRI Classification

The trained deep-learning model classifies an uploaded MRI image into:

* **Tumor**
* **No Tumor**

The API also returns a confidence score associated with the prediction.

---

### 🔥 Explainable AI with Grad-CAM

Instead of treating the model as a black box, the application generates a **Grad-CAM heatmap** showing the regions of the MRI that contributed most strongly to the model's prediction.

```text id="z2f8pj"
             MRI Image
                 │
                 ▼
        ┌─────────────────┐
        │ TensorFlow/Keras│
        │      Model      │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Prediction         Grad-CAM
        │                 │
        │                 ▼
        │          Activation Map
        │                 │
        └────────┬────────┘
                 ▼
       Prediction + Confidence
             + Heatmap
```

The visualization provides an **interpretability signal** for the model's decision. It should not be interpreted as a clinical explanation or diagnostic evidence.

---

### ⚡ FastAPI Inference API

The backend exposes a REST endpoint:

```text
POST /predict
```

It accepts an MRI image and returns the model prediction, confidence score, and Grad-CAM visualization.

---

### 💻 Interactive React Interface

The frontend provides a simple workflow:

```text id="w6a8h4"
Upload MRI
    ↓
Send to API
    ↓
Model Inference
    ↓
Prediction
    ↓
Confidence + Grad-CAM
```

Users can view the prediction and corresponding visualization directly in the browser.

---

### 🌐 CORS Enabled

The FastAPI backend is configured for cross-origin requests, allowing the React frontend to communicate with the inference API during development and deployment.

---

# 🧠 How It Works

The application follows an end-to-end image classification pipeline.

```text id="l7m1fd"
┌───────────────────┐
│    MRI Upload     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Image Preprocessing│
│  Pillow / NumPy   │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ TensorFlow/Keras  │
│   Classification  │
└─────────┬─────────┘
          │
          ├───────────────────┐
          │                   │
          ▼                   ▼
   Classification          Grad-CAM
     Prediction           Activation
          │                   │
          │                   ▼
          │             Heatmap Overlay
          │                   │
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │  React Frontend   │
          │ Prediction +      │
          │ Confidence +      │
          │ Visualization     │
          └───────────────────┘
```

---

# 🔬 Grad-CAM Explainability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize spatial regions that contributed to the model's classification.

The general process is:

```text id="l8i5o1"
Input MRI
   ↓
CNN Feature Maps
   ↓
Target Layer Activations
   ↓
Gradient Calculation
   ↓
Weighted Activation Map
   ↓
Heatmap
   ↓
Overlay on Original MRI
```

This makes it possible to inspect whether the model is focusing on meaningful regions of the input image.

---

# 🏗️ System Architecture

```text id="m5x4xn"
                 ┌───────────────────────┐
                 │      React / Vite     │
                 │       Frontend        │
                 └───────────┬───────────┘
                             │
                        HTTP Request
                             │
                             ▼
                 ┌───────────────────────┐
                 │       FastAPI         │
                 │      /predict         │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ Image Preprocessing   │
                 │ Pillow + NumPy        │
                 └───────────┬───────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │ TensorFlow / Keras    │
                 │      model.h5         │
                 └───────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
               Prediction         Grad-CAM
                    │                 │
                    └────────┬────────┘
                             ▼
                 ┌───────────────────────┐
                 │ Prediction Response   │
                 │ Confidence + Heatmap  │
                 └───────────────────────┘
```

---

# 🛠️ Tech Stack

## Frontend

* **React**
* **Vite**
* **ESLint**

## Backend

* **FastAPI**
* **TensorFlow / Keras**
* **OpenCV**
* **Pillow**
* **NumPy**

## Explainability

* **Grad-CAM**
* CNN activation maps
* Heatmap generation and image overlay

---

# 📁 Project Structure

Use the structure below only if it matches your actual repository:

```text id="m6j7o4"
brain-tumor-detection/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── main.py
│   ├── model.h5
│   ├── requirements.txt
│   └── ...
│
└── README.md
```

> If your actual folder structure differs, update this section to match the repository exactly.

---

# 🚀 Getting Started

## Prerequisites

Make sure you have installed:

* Python 3.9+
* Node.js
* npm

---

## 1. Clone the Repository

```bash id="g6z2nb"
git clone https://github.com/swastikakumari9b44/Braintumor_detection.git
cd Braintumor_detection
```

---

# ⚙️ Backend Setup

Navigate to the backend directory:

```bash id="5x6r3n"
cd backend
```

Create a virtual environment:

```bash id="p7z8cz"
python -m venv venv
```

Activate it.

### Windows

```bash id="w0e7qv"
venv\Scripts\activate
```

### macOS / Linux

```bash id="q6f0jc"
source venv/bin/activate
```

Install dependencies:

```bash id="m1s5fc"
pip install -r requirements.txt
```

Start the FastAPI server:

```bash id="j9k3pa"
uvicorn main:app --reload
```

The API will typically be available at:

```text id="z0xq9e"
http://127.0.0.1:8000
```

FastAPI's interactive API documentation can be accessed at:

```text id="r6h2we"
http://127.0.0.1:8000/docs
```

---

# 💻 Frontend Setup

Open a new terminal and navigate to the frontend:

```bash id="d1p6jh"
cd frontend
```

Install dependencies:

```bash id="k4v9la"
npm install
```

Start the development server:

```bash id="p8e2tz"
npm run dev
```

Open the URL displayed by Vite in your browser.

---

# 🧪 API

## `POST /predict`

Accepts an MRI image for classification.

### Request

```text id="x9q3nf"
Content-Type: multipart/form-data
```

Example form field:

```text id="p7w1aj"
file: <MRI image>
```

### Response

The API returns information corresponding to:

```json id="f2u6yd"
{
  "prediction": "Tumor",
  "confidence": 0.XX,
  "heatmap": "<generated heatmap>"
}
```

> Update the example above to exactly match your backend's actual JSON response.

---

# 📸 Application Preview

## MRI Upload

Add a screenshot of the upload interface here:

```text
screenshots/upload.png
```

![MRI Upload](screenshots/upload.png)

---

## 🧠 Prediction Result

Show the prediction and confidence score:

```text
screenshots/prediction.png
```

![Prediction Result](screenshots/prediction.png)

---

## 🔥 Grad-CAM Visualization

Show the original MRI alongside the Grad-CAM heatmap:

```text
screenshots/gradcam.png
```

![Grad-CAM Visualization](screenshots/gradcam.png)

---

# 📊 Model Performance

This section should contain **actual evaluation results from your trained model**.

| Metric    | Score |
| --------- | ----: |
| Accuracy  | `XX%` |
| Precision | `XX%` |
| Recall    | `XX%` |
| F1 Score  | `XX%` |

### Recommended additional visualizations

* Confusion matrix
* Training vs. validation accuracy
* Training vs. validation loss
* ROC curve
* Example Grad-CAM outputs

> **Do not add numbers unless they come from an actual evaluation of your model.**

---

# ⚠️ Important Disclaimer

This project is an **educational and research-oriented computer vision application**.

It is **not a medical device** and should not be used to diagnose, rule out, or make treatment decisions for brain tumors.

Model predictions and Grad-CAM visualizations depend on the training data, preprocessing pipeline, and model architecture and may contain errors.

---

# 🔮 Future Improvements

Potential improvements include:

* [ ] Multi-class tumor classification
* [ ] Improved model architecture and benchmarking
* [ ] Model evaluation across independent datasets
* [ ] Confidence calibration
* [ ] Additional explainability methods
* [ ] Improved image preprocessing
* [ ] Dockerized deployment
* [ ] Automated backend testing
* [ ] CI/CD with GitHub Actions
* [ ] Cloud deployment
* [ ] Model versioning

---

# 🎯 Project Highlights

This project demonstrates an end-to-end workflow combining:

```text id="u3e1f8"
Computer Vision
      +
Deep Learning
      +
Explainable AI
      +
REST API Development
      +
Modern Frontend Development
```

### What I implemented

* 🧠 Deep-learning based MRI classification
* 🔥 Grad-CAM model interpretability
* ⚡ FastAPI inference backend
* 💻 React/Vite frontend
* 🖼️ Image preprocessing pipeline
* 🌐 Frontend ↔ backend API integration
* 📊 Prediction visualization

---

# 👩‍💻 Author

## Swastika Kumari

AI/ML & Full-Stack Developer

**GitHub:** https://github.com/swastikakumari9b44

**LinkedIn:** www.linkedin.com/in/swastika-kumari-3525b7403



---

## ⭐ Why This Project Matters

Many machine-learning applications stop at:

> **"The model predicted X."**

Brain Tumor Detection goes one step further by attempting to answer:

> **"Which regions of the image influenced the prediction?"**

By combining **deep-learning inference with Grad-CAM visualization**, the project explores how explainability can make computer-vision systems easier to inspect and understand.

---

**Built with React, FastAPI, TensorFlow/Keras, OpenCV, and Grad-CAM.**
