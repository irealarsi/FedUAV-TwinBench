import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2

class SemanticEncoder:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = mobilenet_v2(pretrained=True).features.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def encode(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(input_tensor)
        return features.squeeze().cpu().numpy()  # Returns 1280-d vector

if __name__ == "__main__":
    encoder = SemanticEncoder()
    sample_path = "datasets/drone/raw/VisDrone2019-DET-train/images/0000001_00000_d_0000001.jpg"
    embedding = encoder.encode(sample_path)
    print("Feature vector shape:", embedding.shape)

