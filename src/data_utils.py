import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def load_csv(path):
    df = pd.read_csv(path)
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return X, y

def normalize(X):
    return X / 255.0

def train_val_split(X, y, val_ratio=0.1, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=random_state, shuffle=True
    )
    return X_train, X_val, y_train, y_val

def to_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

def create_dataloader(X_tensor, y_tensor, batch_size=32, shuffle=True):
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader