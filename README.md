# Image Forgery Detection

A deep learning-based web app to detect copy-move image forgeries. Built with **PyTorch**, **Segmentation Models**, and **Streamlit**, it provides both **image-level classification** and **Grad-CAM visualizations** for interpretability.

---

## 🔍 Features

- Detects **forged regions** in images using a hybrid segmentation-classification model.
- Supports **Grad-CAM visualizations** for classifier insights.
- Displays **example images** to demonstrate model predictions.
- Interactive **Streamlit app** interface for easy usage.

---

## 🗂 Repository Structure

ImageForgeryDetection/
├─ app/
│ ├─ app.py # Streamlit app
│ ├─ model.py # Model architecture
│ ├─ infer.py # Inference helper
│ ├─ examples/ # Example images
│ └─ checkpoints/ # Model checkpoints (ignored in git)
|__ Few Data example/
  |-- train_images 
    |-- authentic  # authentic images(.png)
    |-- forged  # forged images(.png)
  |-- train_masks # (masks in .npy)
├─ src/ # Additional scripts
├─ experiments/ # Jupyter notebooks for EDA/training
├─ requirements.txt # Python dependencies
├─ README.md # Project documentation
└─ .gitignore # Ignored files/folders



> ⚠️ **Note:** Large files like model checkpoints and dataset are excluded from GitHub for size constraints.

---

## 📸 Demo

Include screenshots of your Streamlit app here:

![App Screenshot](app/examples/5807.png)


---

## 🚀 Getting Started

### Prerequisites

- Python >= 3.9
- CUDA (optional for GPU inference)
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ImageForgeryDetection.git
cd ImageForgeryDetection

