import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

############################################
# Configuration
############################################
CONFIG = {
    "seed": 42,
    "image_size": (224, 224),
    "batch_size": 16,
    "num_epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "cache_file": "pest_dataset.pkl",
    "model_save": "pest_model_efficientnet.pth"

}

# Set seeds for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################################
# Dataset Loading
############################################
if not os.path.exists(CONFIG["cache_file"]):
    raise RuntimeError(f"Dataset cache file {CONFIG['cache_file']} not found. "
                       f"Please generate it before training.")

print("Loading cached dataset...")
with open(CONFIG["cache_file"], "rb") as f:
    image_bytes, labels = pickle.load(f)

# Create label mappings dynamically from labels
crops = sorted(set(lbl[0] for lbl in labels))
pests = sorted(set(lbl[1] for lbl in labels))
num_crops = max(crops) + 1
num_pests = max(pests) + 1

############################################
# Dataset and Model
############################################
class PestDataset(Dataset):
    def __init__(self, image_bytes, labels, transform=None):
        self.image_bytes = image_bytes
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize(CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_bytes)
    
    def __getitem__(self, idx):
        img = Image.open(BytesIO(self.image_bytes[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        crop_label, pest_label = self.labels[idx]
        return img, torch.tensor(crop_label), torch.tensor(pest_label)

class PestNet(nn.Module):
    def __init__(self, num_crops, num_pests):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        self.crop_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_crops)
        )
        
        self.pest_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_pests)
        )
    
    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.crop_head(x), self.pest_head(x)

############################################
# Training and Evaluation
############################################
def train():
    # Split dataset
    train_data, test_data, train_labels, test_labels = train_test_split(
        image_bytes, labels, test_size=0.2, stratify=[l[0] for l in labels]
    )
    
    # Create datasets
    train_dataset = PestDataset(train_data, train_labels, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]))
    
    test_dataset = PestDataset(test_data, test_labels)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], 
                             shuffle=True, num_workers=CONFIG["num_workers"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                            shuffle=False, num_workers=CONFIG["num_workers"])
    
    # Initialize model
    model = PestNet(num_crops, num_pests).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], 
                          weight_decay=CONFIG["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(CONFIG["num_epochs"]):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, crops, pests in train_loader:
            inputs, crops, pests = inputs.to(device), crops.to(device), pests.to(device)
            
            optimizer.zero_grad()
            crop_preds, pest_preds = model(inputs)
            loss = 0.6*criterion(crop_preds, crops) + 0.4*criterion(pest_preds, pests)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        crop_preds, pest_preds = [], []
        crop_true, pest_true = [], []
        with torch.no_grad():
            for inputs, crops, pests in test_loader:
                inputs = inputs.to(device)
                crops, pests = crops.to(device), pests.to(device)
                
                c_preds, p_preds = model(inputs)
                loss = 0.6*criterion(c_preds, crops) + 0.4*criterion(p_preds, pests)
                val_loss += loss.item()
                
                crop_preds.extend(torch.argmax(c_preds, 1).cpu().numpy())
                pest_preds.extend(torch.argmax(p_preds, 1).cpu().numpy())
                crop_true.extend(crops.cpu().numpy())
                pest_true.extend(pests.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        crop_acc = 100 * accuracy_score(crop_true, crop_preds)
        pest_acc = 100 * accuracy_score(pest_true, pest_preds)
        avg_acc = (crop_acc + pest_acc) / 2
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Crop Acc: {crop_acc:.1f}% | Pest Acc: {pest_acc:.1f}% | Avg: {avg_acc:.1f}%")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), CONFIG["model_save"])
            print(f"Saved new best model with {avg_acc:.1f}% accuracy")
    
    print("\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_acc:.1f}%")

if __name__ == "__main__":
    train()
