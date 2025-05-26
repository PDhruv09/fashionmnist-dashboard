# train_cnn.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn, optim
from models.cnn import CNN
from utils.training import get_data_loaders, train_model, evaluate_model

batch_size = 64
learning_rate = 1e-3
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data_loaders(batch_size=batch_size)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


for epoch in range(num_epochs):
    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%\n")
    scheduler.step()


torch.save(model.state_dict(), "cnn.pth")  # or mlp.pth