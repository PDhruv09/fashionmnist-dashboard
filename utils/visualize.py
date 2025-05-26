# utils/visualize.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import streamlit as st

def plot_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    st.pyplot(plt.gcf())

def show_misclassified(model, dataloader, device, class_names, max_images=9):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(10, 10))

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(images)):
                if preds[i] != labels[i]:
                    plt.subplot(3, 3, images_shown + 1)
                    plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
                    plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
                    plt.axis('off')
                    images_shown += 1
                    if images_shown == max_images:
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        return
