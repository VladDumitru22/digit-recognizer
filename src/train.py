import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import load_csv, normalize, train_val_split, to_tensors, create_dataloader
from model import NeuralNetwork
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

path_train = r'../data/raw/mnist_train.csv'

X, y = load_csv(path=path_train)
X = normalize(X)
X_train, X_val, y_train, y_val = train_val_split(X, y)
X_train_tensor, y_train_tensor = to_tensors(X_train, y_train)
X_val_tensor, y_val_tensor = to_tensors(X_val, y_val)

train_loader = create_dataloader(X_train_tensor, y_train_tensor, batch_size=32)
val_loader = create_dataloader(X_val_tensor, y_val_tensor, batch_size=32)

model = NeuralNetwork(input_shape=28*28, output_shape=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

save_dir = "../models/"
os.makedirs(save_dir, exist_ok=True)

epochs = 10
best_val_acc = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} train loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100*correct/total
    print(f"Epoch {epoch+1} val loss: {val_loss/len(val_loader):.4f}, accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(save_dir, "mnist_model_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Saved new best model with val accuracy: {best_val_acc:.2f}% at {save_path}")
