import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open3d as o3d

# -------------------------------
# 1. Load Point Cloud from File
# -------------------------------
def load_point_cloud(file_path):
    """
    Loads a point cloud from a KITTI .bin file.
    Each file contains (x, y, z, intensity) for every point.
    Returns only the x, y, z values.
    """
    pcd = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pcd[:, :3]


# -------------------------------
# 2. Define PointNet++ Feature Extractor
# -------------------------------
class PointNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(PointNetFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x is expected to be of shape [batch_size, 3, num_points]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # Global feature, shape: [batch_size, 256, 1]
        x = x.view(-1, 256)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x  # Returns feature vector of shape [batch_size, 64]


# -------------------------------
# 3. Define Region Proposal Network (RPN)
# -------------------------------
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.fc_cls = nn.Linear(256, 1)   # Binary classification: object / no object
        self.fc_reg = nn.Linear(256, 7)     # 3D Box regression (x, y, z, h, w, l, yaw)

    def forward(self, x):
        # x: [batch_size, 64, num_points]. In our training we mimic a spatial dim with unsqueeze.
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # Global pooling to reduce the spatial dimension
        x_pooled = torch.max(x, 2, keepdim=False)[0]
        cls_score = torch.sigmoid(self.fc_cls(x_pooled))  # Expected shape: [batch, 1]
        bbox_reg = self.fc_reg(x_pooled)                  # Expected shape: [batch, 7]
        return cls_score, bbox_reg


# -------------------------------
# 4. Define RoI Pooling (if needed)
# -------------------------------
class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super(RoIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, proposals, feature_maps):
        pooled_features = []
        for proposal in proposals:
            pooled_feature = feature_maps[:, :, proposal[0]:proposal[1]]
            pooled_features.append(nn.AdaptiveMaxPool1d(self.output_size)(pooled_feature))
        return torch.stack(pooled_features, dim=0)


# -------------------------------
# 5. Define Loss Function
# -------------------------------
def compute_loss(cls_scores, bbox_reg, labels):
    # Squeeze only the channel dimension so that cls_scores has shape [batch_size]
    cls_loss = nn.BCELoss()(cls_scores.squeeze(1), labels[:, 0])
    bbox_loss = nn.SmoothL1Loss()(bbox_reg, labels[:, 1:])
    return cls_loss + bbox_loss


# -------------------------------
# 6. Training Function
# -------------------------------
def train(model, dataset, epochs=10):
    feature_extractor, rpn = model
    optimizer = optim.Adam(list(feature_extractor.parameters()) + list(rpn.parameters()), lr=0.001)
    
    for epoch in range(epochs):
        for batch in dataset:
            optimizer.zero_grad()
            points, labels = batch  # points: [batch, 3, num_points], labels: [batch, 8]
            features = feature_extractor(points)  # [batch, 64]
            # Mimic spatial dimension for RPN input: [batch, 64, 1]
            features_expanded = features.unsqueeze(2)
            cls_scores, bbox_reg = rpn(features_expanded)
            
            loss = compute_loss(cls_scores, bbox_reg, labels)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")


# -------------------------------
# 7. Load KITTI Dataset
# -------------------------------
def load_kitti_data(dataset_path, num_samples=100):
    """
    Expects dataset_path to be the directory containing .bin files.
    For example: r'C:\\Users\\mbavi\\data_object_velodyne\\training\\velodyne'
    """
    data = []
    files = sorted(os.listdir(dataset_path))[:num_samples]

    for file in files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.isfile(file_path):
            print(f"Warning: File not found {file_path}")
            continue

        points = load_point_cloud(file_path)
        # Create tensor and transpose to shape [1, 3, num_points]
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
        # Dummy labels:
        # For classification: use a value between 0 and 1 (e.g., 1 for object present)
        # For bbox regression: use dummy values (replace with actual bbox labels)
        dummy_cls = torch.tensor([[1.0]])  # shape [1, 1]
        dummy_reg = torch.rand((1, 7))     # shape [1, 7] with values in [0, 1]
        labels = torch.cat([dummy_cls, dummy_reg], dim=1)  # shape [1, 8]
        data.append((points_tensor, labels))

    return data


# -------------------------------
# 8. Inference Function
# -------------------------------
def infer_point_rcnn(point_cloud_file, feature_extractor, rpn):
    point_cloud = load_point_cloud(point_cloud_file)
    point_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    
    features = feature_extractor(point_tensor)
    features_expanded = features.unsqueeze(2)  # Mimic extra dim for RPN
    cls_scores, bbox_reg = rpn(features_expanded)
    
    # Filter detections based on a confidence threshold (e.g., 0.5)
    detected_objects = bbox_reg[cls_scores.squeeze(1) > 0.5]
    return detected_objects


# -------------------------------
# 9. Main Execution
# -------------------------------
if __name__ == "__main__":
    # Update these paths to match your local dataset organization
    training_path = r'C:\Users\mbavi\data_object_velodyne\training\velodyne'
    testing_path = r'C:\Users\mbavi\data_object_velodyne\testing\velodyne\000001.bin'
    
    # Load dataset (using training .bin files)
    dataset = load_kitti_data(training_path, num_samples=100)
    
    # Initialize models
    feature_extractor = PointNetFeatureExtractor()
    rpn = RPN()
    
    # Train the model
    train([feature_extractor, rpn], dataset, epochs=10)
    
    # Run inference on a test point cloud
    detected_objects = infer_point_rcnn(testing_path, feature_extractor, rpn)
    print("Detected Objects:", detected_objects)
