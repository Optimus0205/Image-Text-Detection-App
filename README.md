# 🔍 Image Prediction App — YOLOv11 Object Detection

**An interactive web app that runs a custom-trained YOLOv11 model on any uploaded image and returns annotated detections in real time.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?logo=yolo&logoColor=black)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](#)

[Live Demo](#) · [Report Bug](../../issues) · [Request Feature](../../issues)

---

## 🧭 Overview

Upload an image, hit **Predict**, and watch a custom-trained YOLOv11 model detect and label objects on the spot — bounding boxes, class names, and confidence scores, all rendered side-by-side with the original image. The project ships in **two interchangeable frontends** (Streamlit and Flask) that both call the exact same inference pipeline, so you can pick whichever fits your deployment target.

| 🖼️ Upload Image        | ⚡ Run Inference          | 🎯 View Detections           | 📊 Confidence Scores    |
| ----------------------- | ------------------------- | ----------------------------- | ------------------------ |
| Drag & drop or browse   | YOLOv11 custom `best.pt`  | Annotated bounding boxes      | Per-object confidence %  |

---

## ✨ Features

| Component                  | What it does                                                                                                                                       |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 🖼️ **Image Upload**         | Upload any `.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp` image through a clean, centered file picker                                                     |
| 🤖 **YOLOv11 Inference**    | A custom-trained Ultralytics YOLOv11 model (`best.pt`) detects objects at a **0.50 confidence threshold**                                              |
| 🎯 **Annotated Output**     | Bounding boxes and labels are drawn onto the image using `supervision`'s `BoxAnnotator` and `LabelAnnotator`                                           |
| 📊 **Detection Details**    | Every detection lists its **class name**, **bounding box coordinates**, and **confidence score** in a readable results panel                          |
| 🖥️ **Dual Frontends**       | Choose between a **Streamlit app** (`app.py`) for quick local/cloud demos, or a **Flask REST API + HTML UI** (`index.py` + `templates/index.html`)     |
| 🌐 **CORS-Enabled API**     | The Flask backend exposes a `/predict` POST endpoint with CORS support, so it can be called from any frontend or client                                |

---

## 🖥️ Tech Stack

- **Object Detection:** [Ultralytics YOLOv11](https://docs.ultralytics.com/)
- **Detection Utilities:** [supervision](https://supervision.roboflow.com/) (box & label annotation)
- **Image Processing:** [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/)
- **Frontend (Option 1):** [Streamlit](https://streamlit.io/)
- **Frontend (Option 2):** [Flask](https://flask.palletsprojects.com/) + [Flask-CORS](https://flask-cors.readthedocs.io/) + HTML/CSS/JavaScript
- **Deployment:** [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your model weights

Place your trained `best.pt` file in the project root (see [Notes](#-notes) if it's too large for GitHub).

### 4. Run the app

**Streamlit version:**
```bash
streamlit run app.py
```
Opens automatically at **http://localhost:8501**

**Flask version:**
```bash
python index.py
```
Visit **http://localhost:8000**

---

## 📁 Folder Structure

```
image-prediction-app/
├── app.py                 # Streamlit frontend + UI
├── index.py                # Flask backend (REST API + HTML UI)
├── inference.py             # Shared YOLOv11 inference logic (used by both frontends)
├── requirements.txt          # Python dependencies
├── best.pt                   # Custom-trained YOLOv11 model weights
├── README.md                  # Project documentation
└── templates/
    └── index.html               # Flask HTML frontend
```

---

## ☁️ Deployment

**Streamlit Community Cloud (recommended, free)**

1. Push this repo to GitHub ✅
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app** → select this repo, branch `main`, and main file `app.py`
4. Click **Deploy**

**Flask (self-hosted / any Python host)**

```bash
pip install -r requirements.txt
python index.py
```

---

## 📝 Notes

- Both frontends call the **same** `predict()` function in `inference.py` — model logic never has to be duplicated or kept in sync between the two UIs.
- The model runs at a fixed confidence threshold of `0.50`; adjust the `conf` parameter in `inference.py` to tune sensitivity.
- `best.pt` can exceed GitHub's 100 MB upload limit depending on the model size. If it does, track it with [Git LFS](https://git-lfs.github.com/) or host it externally and download it at app startup.
- Inference runs on **CPU** by default when deployed on Streamlit Community Cloud's free tier — expect slower prediction times than on a local GPU machine.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
<div align="center">

Made with ❤️ using Streamlit

## 🙋‍♂️ Author

**[Optimus0205](https://github.com/Optimus0205)**

⭐ If you found this project useful, consider giving it a star!

</div>

## 🙋 Author

**[Your Name](https://github.com/<your-username>)**

⭐ If you found this project useful, consider giving it a star!
