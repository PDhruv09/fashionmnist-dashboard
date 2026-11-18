# 👕 FashionMNIST Model Comparison & Explainability Dashboard

This interactive project compares deep learning models trained on the FashionMNIST dataset using a clean, modern Streamlit dashboard. It highlights key model insights such as prediction confidence, test accuracy, and misclassified examples — great for showcasing deep learning understanding and model evaluation skills.

🔗 **Live Demo**: [fashion.dhruvkpatel.dev](https://fashion.dhruvkpatel.dev)

---

## 🔍 Features

- Compare **MLP** and **CNN** model predictions
- Interactive visualizations:
  - 🔁 Confusion Matrix
  - ❌ Misclassified Images
  - 🔍 Inspect predictions on any test image
  - 📊 Side-by-side model accuracy comparison
  - 📈 Prediction confidence bar chart
- Responsive, toggled layout with Streamlit

---

## 🧰 Tech Stack

- Python & PyTorch
- Streamlit (frontend)
- Matplotlib & Seaborn
- Scikit-learn

---

## 📁 Data

The data was take from an open dataset, called FashionMINST. here is the like for the repo https://github.com/zalandoresearch/fashion-mnist

---

## 📁 Project Structure

```plaintext
models/       # MLP and CNN architectures
utils/        # Training, evaluation, visualization code
dashboard/    # Streamlit frontend (app.py)
assets/       # (optional) saved images, plots
requirements.txt
README.md
train_mlp.py  # MLP training script
train_cnn.py  # CNN training script
mlp.pth       # Saved trained MLP model
cnn.pth       # Saved trained CNN model
