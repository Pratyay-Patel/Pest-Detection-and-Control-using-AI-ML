import base64
import io
import os
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from flask import Flask, request, render_template_string, Response, url_for
import cv2

app = Flask(__name__)

# ============================
# Configuration and Label Maps
# ============================
CONFIG = {
    "image_size": (224, 224),
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "model_path": "pest_model.pth"  # Ensure this file exists in your working directory
}
# The checkpoint was trained with 7 pest classes.
# For binary display we ignore the actual names and simply invert:
# Previously: if predicted index==0, we showed "Pest Detected"
# Now: if raw prediction is nonzero then we treat it as "Pest Detected"
LABEL_MAPS = {
    "crops": ["Wheat", "Rice", "Corn", "Tomato"],
    "pests": ["No Pest", "Aphid", "Rust", "Planthopper", "Blight", "Whitefly", "Extra"]
}

# ============================
# Image Transformations
# ============================
def get_transforms(mode='val'):
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

transform = get_transforms('val')

# ============================
# Model Architecture (PestResNet)
# Using EfficientNet-B0 as backbone with an adapter to match checkpoint dimensions.
# ============================
class PestResNet(nn.Module):
    def __init__(self, num_crops, num_pests):
        super(PestResNet, self).__init__()
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = 1280  # EfficientNet-B0 outputs 1280 features

        # Adapter layer: converting 1280 -> 2048 to match the checkpoint's head dimensions
        self.adapter = nn.Linear(num_ftrs, 2048)
        
        self.crop_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_crops)
        )
        self.pest_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_pests)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.adapter(x)  # Convert to 2048 features
        return self.crop_head(x), self.pest_head(x)

# ============================
# Load the Model from Checkpoint
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PestResNet(len(LABEL_MAPS["crops"]), len(LABEL_MAPS["pests"])).to(device)
checkpoint = torch.load(CONFIG["model_path"], map_location=device)
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    state_dict = checkpoint["model_state"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=False)
model.eval()

# ============================
# Utility Functions
# ============================
def process_image(url):
    try:
        response = requests.get(url, timeout=10)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        return img.resize(CONFIG["image_size"])
    except Exception as e:
        print(f"Image error: {str(e)}")
        return None

def predict(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, pest_logits = model(tensor)
    pest_probs = torch.softmax(pest_logits, dim=1)
    raw_pred = torch.argmax(pest_probs, dim=1).item()
    confidence = pest_probs.max().item()
    # NEW BINARY MAPPING:
    # If the raw prediction is nonzero, we now treat it as "Pest Detected"
    # Otherwise, if raw_pred == 0, we consider it as "Pest Not Detected"
    if raw_pred == 0:
        pest_pred = 1  # 1 means "Pest Not Detected"
    else:
        pest_pred = 0  # 0 means "Pest Detected"
    return pest_pred, confidence

def get_display_text(pest_pred):
    if pest_pred == 0:
        return "Pest Detected"
    else:
        return "Pest Not Detected"

# ============================
# Navigation Bar (shared across pages)
# ============================
nav_bar = """
<nav class="navbar navbar-expand-lg fixed-top" style="background: transparent; box-shadow: none;">
  <div class="container">
    <div class="collapse navbar-collapse justify-content-center">
      <ul class="navbar-nav" style="gap: 30px;">
        <li class="nav-item">
          <a class="nav-link" href="/">Home</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="/about">About Us</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="/contact">Contact Us</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="/schemes">Government Schemes</a>
        </li>
        <li class="nav-item">
          <a class="nav-link" href="/live">Live Tracking</a>
        </li>
      </ul>
    </div>
  </div>
</nav>
"""

# ============================
# Home Page Template (URL-Based Detection)
# ============================
home_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Pest Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
      html { scroll-behavior: smooth; }
      body { margin: 0; padding: 0; font-family: 'Arial Rounded MT Bold', sans-serif; background: #f4f4f4; }
      #bgVideo { position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; object-fit: cover; filter: brightness(0.6); }
      .main-container { background: rgba(255, 255, 255, 0.95); border-radius: 20px; padding: 2.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(12px); max-width: 800px; margin: 120px auto 2rem auto; }
      .input-card { background: rgba(255, 255, 255, 0.9); border-radius: 15px; padding: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }
      .custom-input { border: 2px solid #1a237e; border-radius: 12px; padding: 1rem; font-size: 1.1rem; transition: all 0.3s ease; }
      .scan-button { background: linear-gradient(45deg, #1a237e, #b71c1c); color: white; padding: 1rem 2.5rem; border: none; border-radius: 12px; font-weight: 600; transition: all 0.3s ease; }
      .result-card { background: rgba(255, 255, 255, 0.95); border-radius: 15px; margin-top: 2rem; padding: 1.5rem; border-left: 5px solid #b71c1c; }
    </style>
</head>
<body>
    {{ nav_bar|safe }}
    <video autoplay muted loop id="bgVideo">
      <source src="https://media.istockphoto.com/id/2019072266/it/video/campo-di-mais-maturo-con-raccolto-e-sfondo-del-cielo-pomeridiano.mp4?s=mp4-640x640-is&k=20&c=YfdtUbhc0F4cAY9hckgVbEKVnixcRLlOP3ZOC04I92s=" type="video/mp4">
    </video>
    <div class="main-container">
        <h1 class="text-center mb-5" style="color: #1a237e;">
            <i class="fas fa-bug"></i> Detect Pests
        </h1>
        <div class="input-card">
            <form method="POST">
                <div class="mb-4">
                    <label class="form-label">Enter Image URL:</label>
                    <input type="text" class="form-control custom-input" name="image_url" placeholder="e.g., https://example.com/crop-image.jpg" required>
                </div>
                <div class="text-center">
                    <button type="submit" class="scan-button">
                        <i class="fas fa-search"></i> Detect Pests
                    </button>
                </div>
            </form>
            {% if result %}
            <div class="result-card mt-4">
                <div class="row">
                    <div class="col-md-5 text-center">
                        <img src="data:image/jpeg;base64,{{ img_base64 }}" class="img-fluid rounded" style="max-height: 200px;">
                    </div>
                    <div class="col-md-7">
                        <h3 class="text-{% if pest_pred == 0 %}danger{% else %}success{% endif %}">
                            {% if pest_pred == 0 %}
                                <i class="fas fa-exclamation-triangle"></i> Pest Detected
                            {% else %}
                                <i class="fas fa-check-circle"></i> Pest Not Detected
                            {% endif %}
                        </h3>
                        <p>
                            {% if pest_pred == 0 %}
                                Warning: Pest presence detected!
                            {% else %}
                                Your crop appears healthy.
                            {% endif %}
                        </p>
                        <div class="mt-3">
                            <h5>Recommended Actions:</h5>
                            <ul class="list-group">
                                {% if pest_pred == 0 %}
                                    <li class="list-group-item">Isolate affected plants</li>
                                    <li class="list-group-item">Apply recommended pesticides</li>
                                    <li class="list-group-item">Consult an agriculture expert</li>
                                {% else %}
                                    <li class="list-group-item">Continue regular monitoring</li>
                                    <li class="list-group-item">Maintain proper irrigation</li>
                                    <li class="list-group-item">Apply preventive treatments</li>
                                {% endif %}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

about_template = """
<!DOCTYPE html>
<html>
<head>
    <title>About Us - Pest Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      html { scroll-behavior: smooth; }
      body { margin: 0; padding: 0; font-family: 'Arial Rounded MT Bold', sans-serif; background: #f4f4f4; }
      #bgVideo { position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; object-fit: cover; filter: brightness(0.6); }
      .container-custom { margin: 120px auto; max-width: 800px; background: linear-gradient(135deg, rgba(26,35,126,0.85), rgba(183,28,28,0.85)); padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: #fff; }
      h1 { text-align: center; margin-bottom: 20px; }
      p { font-size: 1.1rem; line-height: 1.6; }
      ul { list-style: none; padding-left: 0; }
      li { margin-bottom: 10px; font-size: 1.1rem; }
    </style>
</head>
<body>
    {{ nav_bar|safe }}
    <video autoplay muted loop id="bgVideo">
      <source src="https://media.istockphoto.com/id/2019072266/it/video/campo-di-mais-maturo-con-raccolto-e-sfondo-del-cielo-pomeridiano.mp4?s=mp4-640x640-is&k=20&c=YfdtUbhc0F4cAY9hckgVbEKVnixcRLlOP3ZOC04I92s=" type="video/mp4">
    </video>
    <div class="container-custom">
        <h1>About Us</h1>
        <p>We empower farmers with cutting-edge technology to protect crops from pest infestations. Our system leverages advanced machine learning to detect pests early, reducing crop loss and supporting sustainable agriculture.</p>
        <p>By combining innovation with traditional farming wisdom, we strive to offer reliable solutions for a healthy agricultural future.</p>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

contact_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Contact Us - Pest Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      html { scroll-behavior: smooth; }
      body { margin: 0; padding: 0; font-family: 'Arial Rounded MT Bold', sans-serif; background: #f4f4f4; }
      #bgVideo { position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; object-fit: cover; filter: brightness(0.6); }
      .container-custom { margin: 120px auto; max-width: 800px; background: linear-gradient(135deg, rgba(26,35,126,0.85), rgba(183,28,28,0.85)); padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: #fff; }
      h1 { text-align: center; margin-bottom: 20px; }
      p, li { font-size: 1.1rem; line-height: 1.6; }
      ul { list-style: none; padding-left: 0; }
      li { margin-bottom: 10px; }
    </style>
</head>
<body>
    {{ nav_bar|safe }}
    <video autoplay muted loop id="bgVideo">
      <source src="https://media.istockphoto.com/id/2019072266/it/video/campo-di-mais-maturo-con-raccolto-e-sfondo-del-cielo-pomeridiano.mp4?s=mp4-640x640-is&k=20&c=YfdtUbhc0F4cAY9hckgVbEKVnixcRLlOP3ZOC04I92s=" type="video/mp4">
    </video>
    <div class="container-custom">
        <h1>Contact Us</h1>
        <p>For assistance with pest detection, please contact us:</p>
        <ul>
            <li><strong>Agriculture Ministry Helpline:</strong> 1800-180-1551</li>
            <li><strong>Pest Control Helpline:</strong> 1962</li>
            <li><strong>Email:</strong> support@agriculture.gov.in</li>
        </ul>
        <p>Our team is available to provide expert advice and support.</p>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

schemes_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Government Schemes - Pest Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      html { scroll-behavior: smooth; }
      body { margin: 0; padding: 0; font-family: 'Arial Rounded MT Bold', sans-serif; background: #f4f4f4; }
      #bgVideo { position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; object-fit: cover; filter: brightness(0.6); }
      .container-custom { margin: 120px auto; max-width: 800px; background: linear-gradient(135deg, rgba(26,35,126,0.85), rgba(183,28,28,0.85)); padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); color: #fff; }
      h1 { text-align: center; margin-bottom: 20px; }
      p, li { font-size: 1.1rem; line-height: 1.6; }
      ul { list-style: none; padding-left: 0; }
      li { margin-bottom: 10px; }
    </style>
</head>
<body>
    {{ nav_bar|safe }}
    <video autoplay muted loop id="bgVideo">
      <source src="https://media.istockphoto.com/id/2019072266/it/video/campo-di-mais-maturo-con-raccolto-e-sfondo-del-cielo-pomeridiano.mp4?s=mp4-640x640-is&k=20&c=YfdtUbhc0F4cAY9hckgVbEKVnixcRLlOP3ZOC04I92s=" type="video/mp4">
    </video>
    <div class="container-custom">
        <h1>Government Schemes</h1>
        <p>Learn about the various government schemes that support farmers in combating pest infestations and maintaining healthy crops.</p>
        <ul>
            <li><strong>PM-Kisan Scheme:</strong> Direct financial support for modern agricultural practices.</li>
            <li><strong>Rashtriya Krishi Vikas Yojana:</strong> Infrastructure development and innovation in agriculture.</li>
            <li><strong>Soil Health Card Scheme:</strong> Detailed insights for optimal crop management.</li>
            <li><strong>Regional Programs:</strong> Tailored assistance based on local needs.</li>
        </ul>
        <p>These initiatives aim to enhance productivity and secure a sustainable future for agriculture.</p>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

live_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Live Tracking - Pest Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
       html { scroll-behavior: smooth; }
       body { margin: 0; padding: 0; font-family: 'Arial Rounded MT Bold', sans-serif; background: #f4f4f4; }
       #bgVideo { position: fixed; right: 0; bottom: 0; min-width: 100%; min-height: 100%; z-index: -1; object-fit: cover; filter: brightness(0.6); }
       .main-container {
           background: rgba(255, 255, 255, 0.95);
           border-radius: 20px;
           padding: 2.5rem;
           box-shadow: 0 8px 32px rgba(0,0,0,0.1);
           backdrop-filter: blur(12px);
           max-width: 800px;
          margin: 120px auto 2rem auto;
           text-align: center;
       }
    </style>
</head>
<body>
    {{ nav_bar|safe }}
    <video autoplay muted loop id="bgVideo">
      <source src="https://media.istockphoto.com/id/2019072266/it/video/campo-di-mais-maturo-con-raccolto-e-sfondo-del-cielo-pomeridiano.mp4?s=mp4-640x640-is&k=20&c=YfdtUbhc0F4cAY9hckgVbEKVnixcRLlOP3ZOC04I92s=" type="video/mp4">
    </video>
    <div class="main-container">
         <h1 class="mb-4" style="color: #1a237e;">Live Tracking</h1>
         <img src="{{ url_for('video_feed') }}" class="img-fluid rounded" style="max-width:100%;"/>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ============================
# Video Streaming for Live Tracking
# ============================
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_resized = pil_img.resize(CONFIG["image_size"])
        tensor = transform(img_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            _, pest_logits = model(tensor)
        pest_probs = torch.softmax(pest_logits, dim=1)
        raw_pred = torch.argmax(pest_probs).item()
        # Use the same binary mapping as in predict():
        if raw_pred == 0:
            text = "Pest Not Detected"
            color = (0, 255, 0)
        else:
            text = "Pest Detected"
            color = (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ============================
# Flask Routes
# ============================
@app.route("/", methods=["GET", "POST"])
def home():
    context = {
        "result": False,
        "pest_pred": None,
        "img_base64": "",
        "nav_bar": nav_bar
    }
    if request.method == "POST":
        url_input = request.form.get("image_url")
        img = process_image(url_input)
        if img:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            context["img_base64"] = base64.b64encode(buffered.getvalue()).decode()
            pest_pred, _ = predict(img)
            context["pest_pred"] = pest_pred
            context["result"] = True
    return render_template_string(home_template, **context)

@app.route("/about")
def about():
    return render_template_string(about_template, nav_bar=nav_bar)

@app.route("/contact")
def contact():
    return render_template_string(contact_template, nav_bar=nav_bar)

@app.route("/schemes")
def schemes():
    return render_template_string(schemes_template, nav_bar=nav_bar)

@app.route("/live")
def live():
    return render_template_string(live_template, nav_bar=nav_bar)

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


