# -*- coding: utf-8 -*-
"""
Complete Fixed PointPillars Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_PILLARS = 12000
MAX_POINTS_PER_PILLAR = 100
GRID_SIZE = (0, -40, -3, 0.16, 0.16, 1, 512, 448)  # KITTI defaults
FEATURE_SIZE = 64  # Output channels from PFN

# 1. Data Processing ========================================================

def load_point_cloud(file_path: str) -> np.ndarray:
    """Load KITTI point cloud data from binary file"""
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

def create_pillars(points: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert point cloud to pillar format with bounds checking"""
    # Calculate grid coordinates
    x_coords = np.floor((points[:, 0] - GRID_SIZE[0]) / GRID_SIZE[3]).astype(np.int32)
    y_coords = np.floor((points[:, 1] - GRID_SIZE[1]) / GRID_SIZE[4]).astype(np.int32)
    
    # Filter points within grid
    valid_mask = (x_coords >= 0) & (x_coords < GRID_SIZE[6]) & \
                 (y_coords >= 0) & (y_coords < GRID_SIZE[7])
    
    if not np.any(valid_mask):
        return (torch.zeros((MAX_PILLARS, MAX_POINTS_PER_PILLAR, 4), dtype=torch.float32),
                torch.zeros(MAX_PILLARS, dtype=torch.int32),
                torch.zeros((MAX_PILLARS, 3), dtype=torch.int32))
    
    points = points[valid_mask]
    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    
    # Create and constrain pillar indices
    pillar_indices = x_coords * GRID_SIZE[7] + y_coords
    pillar_indices = np.clip(pillar_indices, 0, MAX_PILLARS - 1)
    
    unique_indices, counts = np.unique(pillar_indices, return_counts=True)
    
    # Select top K pillars
    if len(unique_indices) > MAX_PILLARS:
        top_indices = np.argpartition(-counts, MAX_PILLARS)[:MAX_PILLARS]
        selected_indices = unique_indices[top_indices]
        pillar_mask = np.isin(pillar_indices, selected_indices)
        points = points[pillar_mask]
        x_coords = x_coords[pillar_mask]
        y_coords = y_coords[pillar_mask]
        pillar_indices = pillar_indices[pillar_mask]
        unique_indices = selected_indices
    
    # Create output tensors
    features = torch.zeros((MAX_PILLARS, MAX_POINTS_PER_PILLAR, 4), dtype=torch.float32)
    num_points = torch.zeros(MAX_PILLARS, dtype=torch.int32)
    coords = torch.zeros((MAX_PILLARS, 3), dtype=torch.int32)
    
    # Fill pillars
    for i, idx in enumerate(unique_indices):
        mask = pillar_indices == idx
        pillar_points = points[mask]
        n_points = min(pillar_points.shape[0], MAX_POINTS_PER_PILLAR)
        
        features[i, :n_points] = torch.from_numpy(pillar_points[:n_points])
        num_points[i] = n_points
        coords[i, :2] = torch.tensor([x_coords[mask][0], y_coords[mask][0]])
    
    return features, num_points, coords

class KITTIDataset(Dataset):
    def __init__(self, data_dir: str, start_idx: int = 0):
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.bin')])[start_idx:]
        self.data_dir = data_dir
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple:
        pc = load_point_cloud(os.path.join(self.data_dir, self.files[idx]))
        features, num_points, coords = create_pillars(pc)
        
        # Add batch dimension to coords
        batch_coords = torch.zeros((coords.shape[0], 4), dtype=torch.int32)
        batch_coords[:, 1:] = coords
        
        label = torch.randint(0, 3, (1,)).item()  # Dummy label
        return (features, num_points, batch_coords), label

def collate_fn(batch):
    """Custom collate function to handle pillar data"""
    features = torch.cat([item[0][0] for item in batch], dim=0)
    num_points = torch.cat([item[0][1] for item in batch], dim=0)
    
    # Set correct batch indices in coords
    coords = []
    for i, item in enumerate(batch):
        batch_coords = item[0][2]
        batch_coords[:, 0] = i  # Set batch index
        coords.append(batch_coords)
    coords = torch.cat(coords, dim=0)
    
    labels = torch.tensor([item[1] for item in batch])
    return (features, num_points, coords), labels

# 2. Model Architecture ====================================================

class PFNLayer(nn.Module):
    """Pillar Feature Network Layer"""
    def __init__(self, in_channels: int, out_channels: int, last_layer: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.last_layer = last_layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(x)
        if self.last_layer:
            x = torch.max(x, dim=1, keepdim=True)[0]
        return x

class PillarFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pfn_layers = nn.ModuleList([
            PFNLayer(4, 64),
            PFNLayer(64, 64, last_layer=True)
        ])
            
    def forward(self, features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        for pfn in self.pfn_layers:
            features = pfn(features)
        
        # Create pseudo-image with consistent spatial dimensions
        batch_size = coords[:, 0].max().item() + 1
        spatial_features = torch.zeros(
            (batch_size, FEATURE_SIZE, 1, MAX_PILLARS),
            dtype=features.dtype,
            device=features.device)
        
        for b in range(batch_size):
            batch_mask = coords[:, 0] == b
            batch_coords = coords[batch_mask, 1:3]
            batch_features = features[batch_mask].squeeze(1)
            
            # Calculate and constrain indices
            indices = batch_coords[:, 0] * GRID_SIZE[7] + batch_coords[:, 1]
            indices = torch.clamp(indices, 0, MAX_PILLARS - 1).long()
            
            # Scatter features safely
            spatial_features[b, :, 0, indices] = batch_features.T
        
        return spatial_features

class BackboneBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            BackboneBlock(64, 64),
            BackboneBlock(64, 128),
            BackboneBlock(128, 256)
        ])
        self.deblocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(64, 256, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 256, 3, stride=4, padding=1, output_padding=3),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=8, padding=1, output_padding=7),
                nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ups = []
        for i, (block, deblock) in enumerate(zip(self.blocks, self.deblocks)):
            x = block(x)
            ups.append(deblock(x))
        
        # Ensure all feature maps have the same spatial dimensions
        target_size = ups[-1].shape[2:]
        for i in range(len(ups)-1):
            ups[i] = F.interpolate(ups[i], size=target_size, mode='bilinear', align_corners=True)
        
        return torch.cat(ups, dim=1)

class DetectionHead(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(768, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)
        self.conv2 = nn.Conv2d(256, num_classes, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)

class PointPillars(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.pfn = PillarFeatureNet()
        self.backbone = Backbone()
        self.head = DetectionHead(num_classes)
        
    def forward(self, features: torch.Tensor, num_points: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = self.pfn(features, num_points, coords)
        x = self.backbone(x)
        return self.head(x)

# 3. Training Loop =========================================================

def train(model: nn.Module, 
          dataloader: DataLoader, 
          criterion: nn.Module, 
          optimizer: torch.optim.Optimizer,
          epochs: int = 10):
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, ((features, num_points, coords), labels) in enumerate(dataloader):
            # Move data to device
            features = features.to(device)
            num_points = num_points.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, num_points, coords)
            
            # Calculate loss (average over spatial dimensions)
            loss = criterion(outputs.mean(dim=[2, 3]), labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {total_loss/len(dataloader):.4f}")

def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
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
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Configuration
    DATA_DIR = r"C:\Users\mbavi\data_object_velodyne\training\velodyne"
    
    # Create dataset and dataloader
    dataset = KITTIDataset(DATA_DIR, start_idx=934)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    model = PointPillars(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("Starting training...")
    train(model, dataloader, criterion, optimizer, epochs=1)
    
    # Evaluation
    print("Evaluating model...")
    accuracy = evaluate(model, dataloader)
    print(f"Final Accuracy: {accuracy:.2f}%")