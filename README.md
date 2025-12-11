# 👁️ Smart-Eye: AI Assistant for the Visually Impaired

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b?logo=streamlit)

## Project Overview
**Smart-Eye** is a real-time computer vision system designed to assist visually impaired individuals in indoor navigation. The system detects obstacles (such as chairs, people, beds, and tables) through a camera feed and provides immediate **voice feedback (Text-to-Speech)** to warn the user.

The project features a **Full-Stack GUI** (Web Dashboard) for live usage and a separate **Analytics Dashboard** for model evaluation.

---

## Project Pipeline (Step-by-Step)

### 1. Problem Definition
Visually impaired individuals face significant safety challenges navigating indoor environments. Our goal is to build a low-latency, high-accuracy detection system that runs on standard hardware to identify common indoor obstacles and ensure user safety.

### 2. Data Collection
We utilized the **COCO (Common Objects in Context)** dataset standard, specifically focusing on indoor object classes.
* **Source:** COCO / Ultralytics Assets.
* **Target Objects:** Person, Chair, Sofa, Bed, Dining Table, Toilet, TV, Laptop.

### 3. Data Cleaning & Analysis
To ensure model efficiency, we performed the following:
* Filtered the dataset to include only relevant indoor classes (80 -> 9 target classes).
* Verified image integrity and label consistency using `ultralytics` data checking tools.
* Mapped complex class names to simple auditory labels for the TTS system.

### 4. Model Design
We selected the **YOLOv8 Small (yolov8s)** architecture based on **PyTorch**.
* **Reasoning:** It offers the best trade-off between **Inference Speed** (critical for real-time safety) and **Detection Accuracy** (mAP).
* **Framework:** PyTorch (Strictly NO TensorFlow used).

### 5. Model Training (Evidence)
We conducted custom training experiments to optimize performance.
* **Environment:** Local GPU (NVIDIA CUDA).
* **Epochs:** 50.
* **Batch Size:** 8.
* **Optimizer:** AdamW.
* **Results:**
    * **mAP50:** 94.3% (High Accuracy).
    * **Precision:** 87%.
    * **Recall:** 91.5%.
*(Training curves and confusion matrices are available in the `project_results` folder)*.

### 6. Model Testing & Inference
The model was tested on unseen images (Bus, Indoor scenes) and showed robust performance.
* **Inference Speed:** ~40ms per frame (Real-time capable).
* **Evaluation Tool:** We built a custom **Streamlit Dashboard** (`eval.py`) to visualize metrics and test new images dynamically.

### 7. GUI Implementation
We implemented a modern, user-friendly interface:
* **Frontend:** HTML5, CSS3 (Dark Mode Dashboard), and JavaScript.
* **Backend:** **FastAPI** using **WebSockets** for low-latency video streaming.
* **Features:** Live Object Detection, Confidence Score Display, Voice Control (Mute/Unmute), and Activity Logs.

---

## Bonus Features Achieved
1.  **Full Web-Stack Implementation:** We built a complete backend API (`api.py`) and a frontend dashboard (`index.html`) instead of a basic desktop window.
2.  **Interactive Evaluation Dashboard:** Used **Streamlit** to create a GUI for reviewing model training metrics and testing the model offline.
3.  **Real-time Text-to-Speech (TTS):** Integrated Client-side Speech Synthesis for immediate audio feedback.

---

## Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-link-here>
    cd Smart_Eye_Project
    ```

2.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the System (One-Click):**
    Double-click on `run_project.bat`.
    * This will start the **API**, open the **Live Dashboard**, and launch the **Evaluation Tool**.

---

## Team Members & Task Distribution


| Student Name | Student ID | Assigned Task |
| **محمد مدحت محمد عبدالدايم** | 412300377 | **GUI Implementation:** Frontend design (HTML/CSS/JS), Dark Mode Dashboard, and User Experience. |
| **محمد علاء عبد العزيز ذكي** | 412300079 | **Backend API:** FastAPI Server setup, WebSocket implementation, and Real-time data streaming. |
| **أسامه محمد أحمد محمد** | 412300169 | **Documentation & Eval:** Problem Definition, README Documentation, and Streamlit Evaluation Dashboard. |
| **ريناد ايمن رجب فهيم** | 412300040 | **Model Pipeline, Model Training, Testing & Validation:** Data Collection, Preprocessing, and Model Selection (YOLOv8). |
| **منار مجدى السيد شحاته** | 412300103 | **Model Pipeline, Model Training, Testing & Validation:** Training configuration, Hyperparameter tuning, and GPU optimization. |
| **رحمة رضا عبد العليم** | 412300036 | **Model Pipeline, Model Training, Testing & Validation:** Model Testing, Inference on unseen data, and Performance Analysis. |


##  File Structure
* `api.py`: Backend server (FastAPI).
* `index.html`: Frontend interface.
* `eval.py`: Analytics dashboard (Streamlit).
* `run_project.bat`: Launcher script.
* `project_results/`: Contains training evidence (Charts, Weights).
* `models/`: Contains the trained models (`best.pt`).