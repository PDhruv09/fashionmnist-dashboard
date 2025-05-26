# dashboard/app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.mlp import MLP
from models.cnn import CNN
from utils.visualize import plot_confusion_matrix, show_misclassified

st.set_page_config(page_title="FashionMNIST Model Explorer", layout="wide")
st.title("👕 FashionMNIST Model Comparison Dashboard")

# Class labels
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
@st.cache_data
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader

test_loader = load_data()

# Load model
@st.cache_resource
def load_model(model_type):
    if model_type == "MLP":
        model = MLP().to(device)
        model.load_state_dict(torch.load("mlp.pth", map_location=device))
    else:
        model = CNN().to(device)
        model.load_state_dict(torch.load("cnn.pth", map_location=device))
    model.eval()
    return model

# Sidebar
model_choice = st.sidebar.selectbox("Choose Model", ["MLP", "CNN"])
model = load_model(model_choice)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Show Confusion Matrix"):
    st.subheader(f"Confusion Matrix for {model_choice}")
    plot_confusion_matrix(model, test_loader, device, class_names)

if st.sidebar.button("❌ Show Misclassified Images"):
    st.subheader(f"Misclassified Images for {model_choice}")
    show_misclassified(model, test_loader, device, class_names)

st.markdown("---")
st.markdown("Use the sidebar to explore predictions made by different models.")

# 🔍 Inspect a Specific Test Image
st.subheader("🔍 Inspect a Specific Test Image")
index = st.slider("Select test image index", 0, 9999, 0)

# Get one item
dataset = test_loader.dataset
image, label = dataset[index]

# Prepare image
input_image = image.unsqueeze(0).to(device)  # add batch dim
output = model(input_image)
probs = torch.softmax(output, dim=1).cpu().detach().numpy()[0]
pred_label = np.argmax(probs)

unnorm = image * 0.5 + 0.5  # reverse normalization: [-1, 1] ➝ [0, 1]
st.image(unnorm.squeeze().numpy(), caption=f"True Label: {class_names[label]}", width=200)

# Show predictions
st.markdown(f"**Predicted Label:** {class_names[pred_label]}")

# Show confidence bar chart
fig, ax = plt.subplots()
ax.barh(class_names, probs)
ax.set_xlabel("Confidence")
ax.set_xlim(0, 1)
st.pyplot(fig)

# Bar chart comparing models
st.subheader("📊 Model Accuracy Comparison")

model_names = ["MLP", "CNN"]
test_accuracies = [88.29, 91.71]  # Replace with your actual results

fig2, ax2 = plt.subplots()
ax2.bar(model_names, test_accuracies, color=["skyblue", "lightgreen"])
ax2.set_ylim(0, 100)
ax2.set_ylabel("Test Accuracy (%)")
st.pyplot(fig2)
