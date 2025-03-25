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
# 2. Parse KITTI Label File
# -------------------------------
def parse_kitti_label(label_file, object_type="Car"):
    """
    Parses a KITTI label file and returns the label for the first object
    of the specified type. Returns a tensor of shape [1, 8]:
      [cls, x, y, z, h, w, l, rotation_y]
    where cls is 1 if the object is present.
    If no object of that type is found, returns a negative label.
    """
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found {label_file}")
        return torch.tensor([[0.0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if parts[0] == object_type:
            # KITTI order: type, truncated, occluded, alpha,
            # bbox (left, top, right, bottom),
            # dimensions (height, width, length),
            # location (x, y, z), rotation_y
            # We extract: [1, x, y, z, h, w, l, rotation_y]
            # Note: KITTI provides dimensions as (h, w, l)
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            rotation_y = float(parts[14])
            label = torch.tensor([[1.0, x, y, z, h, w, l, rotation_y]], dtype=torch.float32)
            return label
            
    # If no object of the specified type is found, return negative label
    return torch.tensor([[0.0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)


# -------------------------------
# 3. Define PointNet++ Feature Extractor
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
        # x: [batch_size, 3, num_points]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]  # Global feature, shape: [batch, 256, 1]
        x = x.view(-1, 256)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x  # Returns features of shape [batch, 64]


# -------------------------------
# 4. Define Region Proposal Network (RPN)
# -------------------------------
class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.fc_cls = nn.Linear(256, 1)   # Binary classification: object/no-object
        self.fc_reg = nn.Linear(256, 7)     # Regression: [x, y, z, h, w, l, rotation_y]

    def forward(self, x):
        # x: [batch, 64, num_points] (we mimic a spatial dim with unsqueeze)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x_pooled = torch.max(x, 2, keepdim=False)[0]  # Global pooling -> [batch, 256]
        cls_score = torch.sigmoid(self.fc_cls(x_pooled))  # [batch, 1]
        bbox_reg = self.fc_reg(x_pooled)                  # [batch, 7]
        return cls_score, bbox_reg


# -------------------------------
# 5. Define Loss Function
# -------------------------------
def compute_loss(cls_scores, bbox_reg, labels):
    # Squeeze only the channel dimension so that cls_scores is [batch]
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
        epoch_loss = 0.0
        for batch in dataset:
            optimizer.zero_grad()
            points, labels = batch  # points: [1, 3, num_points], labels: [1, 8]
            features = feature_extractor(points)  # [1, 64]
            # Mimic a spatial dimension for RPN: [1, 64, 1]
            features_expanded = features.unsqueeze(2)
            cls_scores, bbox_reg = rpn(features_expanded)
            
            loss = compute_loss(cls_scores, bbox_reg, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataset)}")


# -------------------------------
# 7. Load KITTI Dataset with Actual Labels
# -------------------------------
def load_kitti_data(velodyne_path, label_path, num_samples=100, object_type="Car"):
    """
    Loads KITTI point clouds and corresponding labels.
    Expects:
      - velodyne_path: directory with .bin files.
      - label_path: directory with .txt files.
    """
    data = []
    files = sorted(os.listdir(velodyne_path))[:num_samples]

    for file in files:
        bin_path = os.path.join(velodyne_path, file)
        # Build corresponding label file path (e.g., 000000.txt for 000000.bin)
        label_file = os.path.join(label_path, file.replace('.bin', '.txt'))
        
        if not os.path.isfile(bin_path):
            print(f"Warning: File not found {bin_path}")
            continue
        
        points = load_point_cloud(bin_path)
        # Create tensor and transpose to shape [1, 3, num_points]
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
        # Parse the KITTI label file
        labels = parse_kitti_label(label_file, object_type=object_type)
        data.append((points_tensor, labels))

    return data


# -------------------------------
# 8. Inference Function
# -------------------------------
def infer_point_rcnn(point_cloud_file, feature_extractor, rpn):
    point_cloud = load_point_cloud(point_cloud_file)
    point_tensor = torch.tensor(point_cloud, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    
    features = feature_extractor(point_tensor)
    features_expanded = features.unsqueeze(2)  # Mimic spatial dimension
    cls_scores, bbox_reg = rpn(features_expanded)
    
    # Filter detections based on confidence threshold (e.g., 0.5)
    detected_objects = bbox_reg[cls_scores.squeeze(1) > 0.5]
    return detected_objects


# -------------------------------
# 9. Main Execution
# -------------------------------
if __name__ == "__main__":
    # Update these paths to match your local KITTI dataset organization.
    velodyne_train_path = r'C:\Users\mbavi\data_object_velodyne\training\velodyne'
    label_train_path = r'C:\Users\mbavi\data_object_velodyne\training\label_2'
    testing_path = r'C:\Users\mbavi\data_object_velodyne\testing\velodyne\000001.bin'
    
    # Load dataset (point clouds and corresponding labels)
    dataset = load_kitti_data(velodyne_train_path, label_train_path, num_samples=100, object_type="Car")
    
    # Initialize models
    feature_extractor = PointNetFeatureExtractor()
    rpn = RPN()
    
    # Train the model
    train([feature_extractor, rpn], dataset, epochs=10)
    
    # Run inference on a test point cloud
    detected_objects = infer_point_rcnn(testing_path, feature_extractor, rpn)
    print("Detected Objects:", detected_objects)
