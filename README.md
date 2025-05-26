# FashionMNIST Model Comparison & Explainability Dashboard

This project explores different deep learning models trained on the FashionMNIST dataset, compares their performance, and visualizes key metrics and misclassifications through an interactive dashboard.

## 🔍 Features
- Train & compare MLP, CNN, and ResNet on FashionMNIST
- Visualize:
  - Accuracy, loss, confusion matrix
  - Misclassified images
  - Saliency maps / explanation heatmaps
- Interactive UI with Streamlit

## 🧰 Tech Stack
- PyTorch
- Matplotlib / Seaborn
- Scikit-learn
- Streamlit

## 📁 Structure
models/ # MLP, CNN, ResNet definitions
utils/ # Training, evaluation, plotting helpers
dashboard/ # Streamlit frontend code
notebooks/ # EDA and testing
assets/ # Saved charts, images

## 📦 Setup

```bash
pip install -r requirements.txt