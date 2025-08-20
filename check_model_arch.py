import torch
import torch.nn as nn
import torchvision.models as models

# EfficientNet-B0 architecture
class PestNetEfficient(nn.Module):
    def __init__(self, num_crops=4, num_pests=6):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.crop_head = nn.Linear(1280, num_crops)
        self.pest_head = nn.Linear(1280, num_pests)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.crop_head(x), self.pest_head(x)

# Try loading pest_model.pth
try:
    model = PestNetEfficient()
    state_dict = torch.load("pest_model.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    print("✅ pest_model.pth is EfficientNet-B0")
except Exception as e:
    print("❌ Not EfficientNet-B0:", e)
