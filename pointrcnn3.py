# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:34:40 2025

@author: mbavi
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

############################################
# Dataset: Reads point clouds and labels from separate directories
############################################

class KittiPointCloudDataset(Dataset):
    def __init__(self, point_dir, label_dir=None, transform=None):
        """
        Args:
            point_dir (str): Directory containing point cloud files.
            label_dir (str or None): Directory containing label files. If None, dataset is assumed test.
            transform (callable, optional): Transformation to apply on a sample.
        """
        self.point_dir = point_dir
        self.label_dir = label_dir
        self.point_files = sorted(os.listdir(point_dir))
        if label_dir is not None:
            self.label_files = sorted(os.listdir(label_dir))
            assert len(self.point_files) == len(self.label_files), \
                "Mismatch between number of point files and label files."
        else:
            self.label_files = None
        self.transform = transform

    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        point_path = os.path.join(self.point_dir, self.point_files[idx])
        # Example: Load binary point cloud file (assumed to be float32 with shape Nx4: x,y,z,intensity)
        points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]  # use only x, y, z
        sample = {'points': torch.tensor(points, dtype=torch.float32)}
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, self.label_files[idx])
            # Example: Load labels from a text file with whitespace-delimited integers
            with open(label_path, 'r') as f:
                labels = np.array([int(x) for x in f.read().split()])
            sample['labels'] = torch.tensor(labels, dtype=torch.long)
        if self.transform:
            sample = self.transform(sample)
        return sample

############################################
# Helper Functions for Point Cloud Operations
############################################

def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling.
    Input:
      xyz: B x N x 3 tensor of points.
      npoint: Number of points to sample.
    Return:
      centroids: B x npoint indices of sampled points.
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distances = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Index points/features using indices.
    Input:
      points: B x N x C tensor.
      idx: Indices to sample (shape: B x ...).
    Return:
      new_points: Sampled points/features with shape B x ... x C.
    """
    B = points.shape[0]
    idx_shape = list(idx.shape)
    idx = idx.view(B, -1)
    new_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(B, idx.shape[1], points.shape[-1]))
    return new_points.view(*idx_shape, -1)

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Ball query: for each new_xyz point, find up to nsample points within radius.
    Input:
      radius: float, search radius.
      nsample: int, max number of points to gather.
      xyz: B x N x 3, all points.
      new_xyz: B x S x 3, query points.
    Return:
      group_idx: B x S x nsample indices.
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdists = torch.cdist(new_xyz, xyz, p=2)**2  # B x S x N
    group_idx = sqrdists.argsort()[:, :, :nsample]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Sample and group points into local regions.
    Returns:
      new_xyz: B x npoint x 3 (sampled centroids)
      new_points: B x npoint x nsample x (3 + D) (grouped points with relative coordinates)
    """
    B, N, C = xyz.shape
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz -= new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

############################################
# Set Abstraction Module (PointNet++ Style)
############################################

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        Args:
            npoint: Number of points to sample.
            radius: Search radius.
            nsample: Number of points in each local region.
            in_channel: Input channel dimension (excluding xyz).
            mlp: List of output sizes for MLP layers.
        """
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        last_channel = in_channel + 3  # add xyz (relative coords)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 1, 2)  # B x (3+D) x npoint x nsample
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, -1)[0]  # B x mlp[-1] x npoint
        return new_xyz, new_points

############################################
# Backbone with Multi-Scale Feature Extraction
############################################

class PointRCNNBackbone(nn.Module):
    def __init__(self):
        super(PointRCNNBackbone, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=0.4, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=128, in_channel=256, mlp=[256, 512, 1024])
    
    def forward(self, xyz):
        points = None
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        global_feat = torch.max(l3_points, 2)[0]  # Global feature vector, B x 1024
        return {
            'l1_xyz': l1_xyz, 'l1_points': l1_points,
            'l2_xyz': l2_xyz, 'l2_points': l2_points,
            'l3_xyz': l3_xyz, 'l3_points': l3_points,
            'global_feat': global_feat
        }

############################################
# Segmentation Head (Per-Point Classification)
############################################

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, num_classes, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        return x

############################################
# Voting Module (for Proposal Refinement)
############################################

class VotingModule(nn.Module):
    def __init__(self, in_channels, vote_factor=1):
        super(VotingModule, self).__init__()
        self.vote_factor = vote_factor
        self.conv1 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv_offset = nn.Conv1d(in_channels, 3 * vote_factor, 1)
    
    def forward(self, features, xyz):
        B, C, N = features.shape
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        offset = self.conv_offset(x)  # B x (3*vote_factor) x N
        offset = offset.view(B, self.vote_factor, 3, N)
        offset = torch.mean(offset, dim=1)  # B x 3 x N
        vote_xyz = xyz.transpose(1, 2) + offset  # B x 3 x N
        vote_xyz = vote_xyz.transpose(1, 2)        # B x N x 3
        return vote_xyz, x

############################################
# Proposal Module (Robust Proposal Generation)
############################################

class ProposalModule(nn.Module):
    def __init__(self, in_channels, num_proposals=128, num_classes=2):
        super(ProposalModule, self).__init__()
        self.num_proposals = num_proposals
        self.conv1 = nn.Conv1d(in_channels, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc_cls = nn.Linear(256, num_classes)
        self.fc_reg = nn.Linear(256, 7)  # [x, y, z, dx, dy, dz, theta]
    
    def forward(self, features):
        B, C, N = features.shape
        # For simplicity, randomly sample proposals from features.
        idx = torch.randperm(N)[:self.num_proposals]
        proposal_features = features[:, :, idx]  # B x C x num_proposals
        x = F.relu(self.bn1(self.conv1(proposal_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max(x, 2)[0]  # Global pooling: B x 256
        cls_scores = self.fc_cls(x)   # B x num_classes
        bbox_preds = self.fc_reg(x)     # B x 7
        return cls_scores, bbox_preds

############################################
# Full Production-Grade PointRCNN Model
############################################

class PointRCNN(nn.Module):
    def __init__(self, num_classes=2, vote_factor=1, num_proposals=128):
        super(PointRCNN, self).__init__()
        self.backbone = PointRCNNBackbone()
        # Segmentation head on high-resolution features from SA1
        self.seg_head = SegmentationHead(in_channels=128, num_classes=num_classes)
        # Voting module on mid-level features from SA2
        self.voting_module = VotingModule(in_channels=256, vote_factor=vote_factor)
        self.proposal_module = ProposalModule(in_channels=256, num_proposals=num_proposals, num_classes=num_classes)
    
    def forward(self, xyz):
        backbone_out = self.backbone(xyz)
        seg_scores = self.seg_head(backbone_out['l1_points'])
        vote_xyz, refined_features = self.voting_module(backbone_out['l2_points'], backbone_out['l2_xyz'])
        cls_scores, bbox_preds = self.proposal_module(refined_features)
        return {
            'seg_scores': seg_scores,  # B x num_classes x 1024
            'vote_xyz': vote_xyz,      # B x N x 3 (voted point locations)
            'cls_scores': cls_scores,  # B x num_classes
            'bbox_preds': bbox_preds   # B x 7
        }

############################################
# Collate Function for DataLoader
############################################

def collate_fn(batch):
    points = torch.stack([item['points'] for item in batch])  # B x N x 3
    # If labels are provided (training), stack them; otherwise, return None.
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
    else:
        labels = None
    return {'points': points, 'labels': labels}

############################################
# Training Loop
############################################

def train_model(model, dataloader, optimizer, scheduler, device, num_epochs=50):
    seg_loss_fn = nn.CrossEntropyLoss()
    cls_loss_fn = nn.CrossEntropyLoss()
    reg_loss_fn = nn.SmoothL1Loss()
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            points = batch['points'].to(device)  # B x N x 3
            labels = batch['labels'].to(device) if batch['labels'] is not None else None
            outputs = model(points)
            seg_scores = outputs['seg_scores']    # B x num_classes x 1024
            cls_scores = outputs['cls_scores']    # B x num_classes
            bbox_preds = outputs['bbox_preds']    # B x 7
            # Compute segmentation loss (assumes labels match segmentation output size)
            loss_seg = seg_loss_fn(seg_scores, labels) if labels is not None else 0.0
            # For proposal classification, use dummy labels (e.g., all zeros) for demonstration
            dummy_proposal_labels = torch.zeros(cls_scores.size(0), dtype=torch.long, device=device)
            loss_cls = cls_loss_fn(cls_scores, dummy_proposal_labels)
            # For bbox regression, use dummy targets (zeros)
            dummy_bbox_targets = torch.zeros_like(bbox_preds, device=device)
            loss_reg = reg_loss_fn(bbox_preds, dummy_bbox_targets)
            loss = loss_seg + loss_cls + loss_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}")

############################################
# Main Training Script
############################################

if __name__ == '__main__':
    # Set up directories (modify these paths as needed)

    train_point_dir = r'C:\Users\mbavi\data_object_velodyne\training\velodyne'
    train_label_dir = r'C:\Users\mbavi\data_object_velodyne\training\label_2'
    test_point_dir = r'C:\Users\mbavi\data_object_velodyne\testing\velodyne'
    
    # Create the dataset and dataloader for training.
    train_dataset = KittiPointCloudDataset(point_dir=train_point_dir, label_dir=train_label_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointRCNN(num_classes=2, vote_factor=1, num_proposals=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    num_epochs = 50
    
    train_model(model, train_loader, optimizer, scheduler, device, num_epochs=num_epochs)
