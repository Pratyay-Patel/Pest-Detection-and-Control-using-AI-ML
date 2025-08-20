import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import cv2
import numpy as np
import requests
from io import BytesIO

# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "image_size": (224, 224),
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "model_save": os.path.join(BASE_DIR, "pest_model.pth"),
    "image_folder": os.path.join(BASE_DIR, "dataset_images"),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "google_api_key": "AIzaSyBfEDmAMsBamVUIt1wg3SvV8QG1M6HmSiU",    # Replace with your Google API key
    "google_cse_id": "52b8c798098b94ce3",        # Replace with your Google CSE ID
    "max_augment_images": 100
}

# ============================
# Label Maps
# ============================
LABEL_MAPS = {
    "crops": ["Wheat", "Rice", "Corn", "Tomato"],
    "pests": ["No Pest", "Aphid", "Rust", "Planthopper", "Blight", "Whitefly"]
}

crop2idx = {crop: i for i, crop in enumerate(LABEL_MAPS["crops"])}
pest2idx = {pest: i for i, pest in enumerate(LABEL_MAPS["pests"])}

# ============================
# Model Architecture
# ============================
class PestResNet(nn.Module):
    def __init__(self, num_crops, num_pests):
        super().__init__()
        # Use the default ResNet50 backbone from torchvision
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        layers = list(base.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = base.fc.in_features

        self.crop_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_crops)
        )

        self.pest_head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_pests)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return self.crop_head(x), self.pest_head(x)

# ============================
# Data Augmentation
# ============================
def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(CONFIG["image_size"]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["normalize_mean"], CONFIG["normalize_std"])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(CONFIG["image_size"]),
            transforms.CenterCrop(CONFIG["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(CONFIG["normalize_mean"], CONFIG["normalize_std"])
        ])

# ============================
# Google Image Collector
# ============================
class GoogleImageCollector:
    def __init__(self):
        self.api_key = CONFIG["google_api_key"]
        self.cse_id = CONFIG["google_cse_id"]
        self.base_url = "https://www.googleapis.com/customsearch/v1"

    def search_images(self, query, num=10):
        try:
            response = requests.get(self.base_url, params={
                'q': query,
                'cx': self.cse_id,
                'key': self.api_key,
                'searchType': 'image',
                'num': num
            }, timeout=10)
            return response.json().get('items', [])
        except Exception as e:
            print(f"API Error for query '{query}': {e}")
            return []

    def download_dataset(self):
        os.makedirs(CONFIG["image_folder"], exist_ok=True)
        for crop in LABEL_MAPS["crops"]:
            for pest in LABEL_MAPS["pests"]:
                # Use different query if no pest is present
                query = f"healthy {crop} plant" if pest == "No Pest" else f"{pest} on {crop}"
                print(f"Searching: {query}")
                results = self.search_images(query, CONFIG["max_augment_images"])
                if not results:
                    print(f"No results found for query: {query}")
                for i, item in enumerate(results):
                    try:
                        img_data = requests.get(item['link'], timeout=15).content
                        img = Image.open(BytesIO(img_data)).convert('RGB')
                        filename = f"{crop}_{pest}_{int(time.time())}_{i}.jpg"
                        img.save(os.path.join(CONFIG["image_folder"], filename))
                    except Exception as e:
                        print(f"Failed to download {item.get('link','')}: {e}")

# ============================
# Dataset Class
# ============================
class PestDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform or get_transforms()
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        if not os.path.exists(self.image_folder):
            return samples
        for fname in os.listdir(self.image_folder):
            if fname.startswith('.'):
                continue
            parts = fname.split('_')
            if len(parts) < 2:
                continue
            try:
                crop, pest = parts[0], parts[1]
                samples.append((fname, crop2idx[crop], pest2idx[pest]))
            except KeyError:
                continue
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, crop_idx, pest_idx = self.samples[idx]
        img_path = os.path.join(self.image_folder, fname)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', CONFIG["image_size"], (255, 255, 255))
        return self.transform(img), torch.tensor(crop_idx), torch.tensor(pest_idx)

# ============================
# Training Function
# ============================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Initializing training...")

    # Data collection from Google
    collector = GoogleImageCollector()
    collector.download_dataset()

    # Load dataset
    full_dataset = PestDataset(CONFIG["image_folder"], transform=get_transforms('train'))
    # If no images were downloaded, create dummy images
    if len(full_dataset) == 0:
        print("No images found in the dataset folder after downloading.")
        print("Creating dummy images for training...")
        os.makedirs(CONFIG["image_folder"], exist_ok=True)
        num_dummy = 5  # number of dummy images per crop-pest combination
        for crop in LABEL_MAPS["crops"]:
            for pest in LABEL_MAPS["pests"]:
                for i in range(num_dummy):
                    # Create a random noise image
                    dummy_array = np.uint8(np.random.rand(CONFIG["image_size"][0],
                                                            CONFIG["image_size"][1],
                                                            3) * 255)
                    dummy_img = Image.fromarray(dummy_array)
                    dummy_filename = f"{crop}_{pest}_dummy_{int(time.time())}_{i}.jpg"
                    dummy_img.save(os.path.join(CONFIG["image_folder"], dummy_filename))
        # Reload dataset after creating dummy images
        full_dataset = PestDataset(CONFIG["image_folder"], transform=get_transforms('train'))

    print(f"Total images found: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    # Use validation transforms for the validation split
    val_dataset.dataset.transform = get_transforms('val')

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

    # Model setup
    model = PestResNet(len(crop2idx), len(pest2idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, crop_labels, pest_labels in train_loader:
            inputs, crop_labels, pest_labels = inputs.to(device), crop_labels.to(device), pest_labels.to(device)
            
            optimizer.zero_grad()
            crop_preds, pest_preds = model(inputs)
            loss = 0.7 * criterion(crop_preds, crop_labels) + 0.3 * criterion(pest_preds, pest_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += crop_labels.size(0)
            correct += (crop_preds.argmax(1) == crop_labels).sum().item()

        # Validation step
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, crop_labels, pest_labels in val_loader:
                inputs, crop_labels, pest_labels = inputs.to(device), crop_labels.to(device), pest_labels.to(device)
                crop_preds, pest_preds = model(inputs)
                loss = 0.7 * criterion(crop_preds, crop_labels) + 0.3 * criterion(pest_preds, pest_labels)
                val_loss += loss.item()
                val_correct += (crop_preds.argmax(1) == crop_labels).sum().item()

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'crop2idx': crop2idx,
                'pest2idx': pest2idx,
                'config': CONFIG
            }, CONFIG["model_save"])
            print(f"Saved best model at epoch {epoch+1}")

        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {val_correct/len(val_dataset):.4f}\n")

# ============================
# Pest Detector
# ============================
class PestDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PestResNet(len(crop2idx), len(pest2idx)).to(self.device)
        
        # Load the checkpoint (assumed valid as per the main check)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.transform = get_transforms('val')
        self.search = GoogleImageCollector()

    def predict(self, image):
        # Convert BGR to RGB and prepare tensor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)
        tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            crop_logits, pest_logits = self.model(tensor_img)
        
        crop_probs = torch.softmax(crop_logits, dim=1)
        pest_probs = torch.softmax(pest_logits, dim=1)
        
        return {
            'crop': {
                'class': LABEL_MAPS["crops"][crop_probs.argmax().item()],
                'confidence': crop_probs.max().item()
            },
            'pest': {
                'class': LABEL_MAPS["pests"][pest_probs.argmax().item()],
                'confidence': pest_probs.max().item()
            }
        }

    def run_realtime(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Unable to open camera.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            prediction = self.predict(frame)
            
            if prediction['pest']['confidence'] > 0.6 and prediction['crop']['confidence'] > 0.6:
                text = f"{prediction['crop']['class']} - {prediction['pest']['class']}"
                color = (0, 0, 255) if prediction['pest']['class'] != "No Pest" else (0, 255, 0)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Pest Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    model_path = CONFIG["model_save"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
  
    if os.path.exists(model_path):
        try:
            temp_model = PestResNet(len(crop2idx), len(pest2idx)).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            temp_model.load_state_dict(checkpoint['model_state'])
            print("Found valid existing model.")
        except Exception as e:
            print("Invalid model file detected, retraining...", e)
            os.remove(model_path)
            train_model()
    else:
        print("No model found, starting training...")
        train_model()

    # Run real-time detection
    detector = PestDetector(model_path)
    detector.run_realtime()

