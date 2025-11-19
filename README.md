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



