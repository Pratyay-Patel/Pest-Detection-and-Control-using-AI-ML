import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import requests
from io import BytesIO

# ============================
# Configuration
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "image_size": (224, 224),
    "model_save": os.path.join(BASE_DIR, "pest_model.pth"),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
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

# ============================
# Transforms
# ============================
def get_transforms():
    return transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.CenterCrop(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["normalize_mean"], CONFIG["normalize_std"])
    ])

# ============================
# Pest Detector
# ============================
class PestDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PestNet(len(crop2idx), len(pest2idx)).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.transform = get_transforms()

    def predict(self, image):
        """Predict from OpenCV image (BGR)."""
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

    def predict_from_url(self, url):
        """Download image from URL and predict."""
        response = requests.get(url, timeout=15)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        tensor_img = self.transform(img).unsqueeze(0).to(self.device)

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
        """Run webcam pest detection."""
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
# Main
# ============================
if __name__ == "__main__":
    model_path = CONFIG["model_save"]

    if os.path.exists(model_path):
        print("Found model, loading...")
        detector = PestDetector(model_path)

        # Example: Predict from a URL
        # url = "https://example.com/sample.jpg"
        # print(detector.predict_from_url(url))

        detector.run_realtime()
    else:
        print("No model found. Please train first.")
