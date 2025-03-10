import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset

# Load KITTI 3D Point Cloud Data
def load_kitti_point_cloud(file_path):
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# Convert Point Cloud to Pillars
def point_cloud_to_pillars(point_cloud, max_pillars=12000, max_points_per_pillar=100):
    pillars = np.zeros((max_pillars, max_points_per_pillar, 4), dtype=np.float32)
    num_pillars = min(len(point_cloud), max_pillars)
    for i in range(num_pillars):
        points = point_cloud[i * max_points_per_pillar:(i + 1) * max_points_per_pillar]
        pillars[i, :len(points), :] = points
    return torch.tensor(pillars, dtype=torch.float32)

# KITTI Dataset Class
class KITTIDataset(Dataset):
    def __init__(self, data_folder):
        self.data_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.bin')]
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        point_cloud = load_kitti_point_cloud(self.data_files[idx])
        pillars = point_cloud_to_pillars(point_cloud)
        label = torch.randint(0, 3, (1,))  # Dummy labels for classification
        return pillars.permute(0, 2, 1), label

# Pillar Feature Net
class PillarFeatureNet(nn.Module):
    def __init__(self, num_input_features=4, num_output_features=64):
        super(PillarFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(num_input_features, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, num_output_features, kernel_size=1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, features, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

# Backbone Network
class BackboneNet(nn.Module):
    def __init__(self, input_channels=64, output_channels=128):
        super(BackboneNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, output_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)  # Flatten before DetectionHead
        return x

# Detection Head
class DetectionHead(nn.Module):
    def __init__(self, input_channels=128 * 1024, num_classes=3):
        super(DetectionHead, self).__init__()
        self.fc1 = nn.Linear(input_channels, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Full PointPillars Model
class PointPillarsNet(nn.Module):
    def __init__(self, num_input_features=4, num_classes=3):
        super(PointPillarsNet, self).__init__()
        self.pillar_feature_net = PillarFeatureNet(num_input_features)
        self.backbone = BackboneNet(input_channels=64)
        self.detection_head = DetectionHead(input_channels=128 * 1024, num_classes=num_classes)
        
    def forward(self, x):
        x = self.pillar_feature_net(x)
        x = x.unsqueeze(-1)  # Reshape to (batch, channels, num_points, 1) for Conv2d
        x = self.backbone(x)
        x = self.detection_head(x)
        return x

# Training and Testing the Model
def train_model(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Load Dataset and Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_folder = "path_to_kitti_dataset"
dataset = KITTIDataset(data_folder)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = PointPillarsNet(num_input_features=4, num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer, epochs=5)
test_model(model, dataloader)
