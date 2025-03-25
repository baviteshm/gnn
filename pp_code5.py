# -*- coding: utf-8 -*-
"""
Complete PointPillars Model Evaluation Script
"""

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants (must match training configuration)
MAX_PILLARS = 12000
MAX_POINTS_PER_PILLAR = 100
GRID_SIZE = (0, -40, -3, 0.16, 0.16, 1, 512, 448)  # KITTI defaults
FEATURE_SIZE = 64
NUM_CLASSES = 3

# 1. Model Architecture (must match training exactly)
class PFNLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, last_layer: bool = False):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.norm = torch.nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.last_layer = last_layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = torch.nn.functional.relu(x)
        if self.last_layer:
            x = torch.max(x, dim=1, keepdim=True)[0]
        return x

class PillarFeatureNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = torch.nn.ModuleList([
            PFNLayer(4, 64),
            PFNLayer(64, 64, last_layer=True)
        ])
            
    def forward(self, features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        for pfn in self.pfn_layers:
            features = pfn(features)
        
        batch_size = coords[:, 0].max().item() + 1
        spatial_features = torch.zeros(
            (batch_size, FEATURE_SIZE, 1, MAX_PILLARS),
            dtype=features.dtype,
            device=features.device)
        
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            batch_coords = coords[batch_mask, 1:3]
            batch_features = features[batch_mask].squeeze(1)
            
            indices = batch_coords[:, 0] * GRID_SIZE[7] + batch_coords[:, 1]
            indices = torch.clamp(indices, 0, MAX_PILLARS - 1).long()
            spatial_features[b, :, 0, indices] = batch_features.T
        
        return spatial_features

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
                torch.nn.ReLU()
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
                torch.nn.ReLU()
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, stride=2, padding=1),
                torch.nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                torch.nn.ReLU()
            )
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

class DetectionHead(torch.nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)
        self.conv2 = torch.nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)

class PointPillars(torch.nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.backbone = Backbone()
        self.head = DetectionHead(num_classes)
        
    def forward(self, features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.pfn(features, num_points, coords)
        features = self.backbone(x)
        return self.head(features[-1])

# 2. Data Loading Functions
def load_point_cloud(file_path: str) -> np.ndarray:
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def create_pillars(points: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_coords = np.floor((points[:, 0] - GRID_SIZE[0]) / GRID_SIZE[3]).astype(np.int32)
    y_coords = np.floor((points[:, 1] - GRID_SIZE[1]) / GRID_SIZE[4]).astype(np.int32)
    
    valid_mask = (x_coords >= 0) & (x_coords < GRID_SIZE[6]) & (y_coords >= 0) & (y_coords < GRID_SIZE[7])
    points = points[valid_mask]
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    pillar_indices = x_coords * GRID_SIZE[7] + y_coords
    pillar_indices = np.clip(pillar_indices, 0, MAX_PILLARS - 1)
    
    features = torch.zeros((MAX_PILLARS, MAX_POINTS_PER_PILLAR, 4), dtype=torch.float32)
    num_points = torch.zeros(MAX_PILLARS, dtype=torch.int32)
    coords = torch.zeros((MAX_PILLARS, 3), dtype=torch.int32)
    
    unique_indices = np.unique(pillar_indices)
    for i, idx in enumerate(unique_indices):
        mask = pillar_indices == idx
        pillar_points = points[mask]
        n_points = min(pillar_points.shape[0], MAX_POINTS_PER_PILLAR)
        features[i, :n_points] = torch.from_numpy(pillar_points[:n_points])
        num_points[i] = n_points
        coords[i, :2] = torch.tensor([x_coords[mask][0], y_coords[mask][0]])
    
    return features, num_points, coords

class KITTIDataset(Dataset):
    def __init__(self, data_dir: str):
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])
        self.data_dir = data_dir
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple:
        pc = load_point_cloud(os.path.join(self.data_dir, self.files[idx]))
        features, num_points, coords = create_pillars(pc)
        batch_coords = torch.zeros((coords.shape[0], 4), dtype=torch.int32)
        batch_coords[:, 1:] = coords
        label = torch.randint(0, NUM_CLASSES, (1,)).item()
        return (features, num_points, batch_coords), label

def collate_fn(batch):
    features = torch.cat([item[0][0] for item in batch], dim=0)
    num_points = torch.cat([item[0][1] for item in batch], dim=0)
    coords = []
    for i, item in enumerate(batch):
        batch_coords = item[0][2]
        batch_coords[:, 0] = i
        coords.append(batch_coords)
    coords = torch.cat(coords, dim=0)
    labels = torch.tensor([item[1] for item in batch])
    return (features, num_points, coords), labels

# 3. Model Loading and Evaluation
def load_pretrained_model(model_path: str) -> torch.nn.Module:
    model = PointPillars(num_classes=NUM_CLASSES).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'module.' prefix if model was saved with DataParallel
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume the file is just the state_dict directly
            model.load_state_dict(checkpoint)
            
        model.eval()
        print(f"Successfully loaded pretrained model from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Model architecture must exactly match the saved weights")
        if 'checkpoint' in locals():
            print("Keys in checkpoint:", list(checkpoint.keys()))
            print("Model expects:", model.state_dict().keys())
        raise

def evaluate(model: torch.nn.Module, data_dir: str, batch_size: int = 4):
    dataset = KITTIDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for (features, num_points, coords), labels in dataloader:
            features = features.to(device)
            num_points = num_points.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            
            outputs = model(features, num_points, coords)
            _, predicted = torch.max(outputs.mean(dim=[2, 3]), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy
# In your main execution block:
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"C:\Users\mbavi\pointpillar_7728.pth"
    DATA_DIR = r"C:\Users\mbavi\data_object_velodyne\training\velodyne"
    
    try:
        # Load model with proper checkpoint handling
        model = load_pretrained_model(MODEL_PATH)
        
        # Evaluation
        accuracy = evaluate(model, DATA_DIR)
        
        # Sample prediction
        sample_input, _ = next(iter(DataLoader(
            KITTIDataset(DATA_DIR), 
            batch_size=1, 
            collate_fn=collate_fn
        )))
        with torch.no_grad():
            sample_output = model(
                sample_input[0].to(device),
                sample_input[1].to(device),
                sample_input[2].to(device)
            )
            print(f"\nSample output shape: {sample_output.shape}")
            print(f"Sample prediction: {torch.argmax(sample_output.mean(dim=[2,3]))}")
            
    except Exception as e:
        print(f"Failed to run evaluation: {str(e)}")
        print("Please verify:")
        print("1. The model architecture matches the checkpoint")
        print("2. The checkpoint file is not corrupted")
        print("3. All required files are in the correct paths")
