# analyze_model.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import CNN
from utils.visualize import plot_confusion_matrix, show_misclassified

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Change to MLP if analyzing that
model = CNN().to(device)
model.load_state_dict(torch.load("cnn.pth"))  # Save model after training

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

plot_confusion_matrix(model, test_loader, device, class_names)
show_misclassified(model, test_loader, device, class_names)
