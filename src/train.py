import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
try:
    import config
    from dataset import EmotionDataset
    from model import CNNLSTMModel
except ImportError:
    from . import config
    from .dataset import EmotionDataset
    from .model import CNNLSTMModel

def train_model():
    # Load Data
    print("Loading data...")
    X_path = os.path.join(config.PROCESSED_DATA_DIR, "X.npy")
    y_path = os.path.join(config.PROCESSED_DATA_DIR, "y.npy")
    
    if not os.path.exists(X_path):
        print("Data not found. Run preprocess.py first.")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=42)
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Datasets
    train_dataset = EmotionDataset(X_train, y_train)
    test_dataset = EmotionDataset(X_test, y_test)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model
    model = CNNLSTMModel(num_classes=len(config.EMOTIONS)).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "best_model.pth"))
            print("Saved best model.")
            
    print("Training Complete.")

if __name__ == "__main__":
    train_model()
