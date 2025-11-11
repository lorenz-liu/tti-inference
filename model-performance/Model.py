"""
Neural Network Models for TTI Detection

This module contains the neural network architectures used for Tool-Tissue Interaction (TTI) 
detection. It includes both ResNet-based and Vision Transformer (ViT) based classifiers
that process 5-channel input data (RGB + Depth + Mask).

Classes:
    ROIClassifier: ResNet18-based classifier with 5-channel input
    ROIClassifierNoDepth: ResNet18-based classifier with 4-channel input (no depth)
    ROIClassifierViT: Vision Transformer-based classifier with 5-channel input
    ROIClassifierViTNoDepth: Vision Transformer-based classifier with 4-channel input
    AutoEncoder: Autoencoder for feature learning (experimental)

All classifiers output binary classification (TTI vs No-TTI) using sigmoid activation.
"""

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import ViTModel, ViTConfig
import os

class ROIClassifier(nn.Module):
    """
    ResNet18-based classifier for TTI detection with 5-channel input.
    
    This model processes 5-channel input data (RGB + Depth + Mask) using a ResNet18
    backbone. It includes channel reduction layers to convert 5-channel input to
    3-channel input for the ResNet backbone.
    
    Args:
        num_hoi_classes (int): Number of output classes (typically 2 for binary classification)
    
    Architecture:
        - Channel reduction: 5 -> 4 -> 3 channels
        - Backbone: ResNet18 (pretrained)
        - Classifier: Linear layer with sigmoid activation
        - Output: Binary classification (TTI vs No-TTI)
    """
    
    def __init__(self, num_hoi_classes):
        super().__init__()
      
        # Channel reduction layers
        self.first = nn.Conv2d(5, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove classification head
        
        # Classification head
        self.fc = nn.Linear(512, num_hoi_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_hoi_classes)
        """
        # Channel reduction: 5 -> 4 -> 3 channels
        x = self.first(x)
        x = self.pre_conv(x)         
        
        # Extract features using ResNet backbone
        features = self.backbone(x)   
        
        # Classification with sigmoid activation
        out = F.sigmoid(self.fc(features))
        return out
    
    

class ROIClassifierNoDepth(nn.Module):
    def __init__(self, num_hoi_classes):
        super().__init__()
      
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.fc = nn.Linear(512, num_hoi_classes)
        
    def forward(self, x):
        
        x = self.pre_conv(x)         
        features = self.backbone(x)   
        out = F.sigmoid(self.fc(features))
        return out


class ROIClassifierViT(nn.Module):
    """
    Vision Transformer-based classifier for TTI detection with 5-channel input.
    
    This model processes 5-channel input data (RGB + Depth + Mask) using a Vision Transformer
    (ViT) backbone. It includes channel reduction layers to convert 5-channel input to
    3-channel input for the ViT backbone.
    
    Args:
        num_hoi_classes (int): Number of output classes (typically 2 for binary classification)
    
    Architecture:
        - Channel reduction: 5 -> 4 -> 3 channels
        - Backbone: Vision Transformer (ViT-Base, patch16-224)
        - Classifier: Linear layer with sigmoid activation
        - Output: Binary classification (TTI vs No-TTI)
        - Features: 768-dimensional embeddings from [CLS] token
    """
    
    def __init__(self, num_hoi_classes):
        super().__init__()
        
        # Channel reduction layers - same as ResNet version
        self.first = nn.Conv2d(5, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Vision Transformer backbone - load locally to avoid internet
        vit_path = './pretrained_models/vit-base-patch16-224'
        self.backbone = ViTModel.from_pretrained(vit_path, local_files_only=True)
        
        # Classification head - ViT outputs 768-dim features
        self.fc = nn.Linear(768, num_hoi_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 5, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_hoi_classes)
        """
        # Channel reduction: 5 -> 4 -> 3 channels
        x = self.first(x)
        x = self.pre_conv(x)
        
        # ViT expects inputs in range [0, 1] and specific format
        # Extract features using ViT backbone
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # Use [CLS] token representation
        
        # Classification with sigmoid activation
        out = F.sigmoid(self.fc(features))
        return out


class ROIClassifierViTNoDepth(nn.Module):
    def __init__(self, num_hoi_classes):
        super().__init__()
        
        # Channel reduction layer - same as ResNet version
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Vision Transformer backbone
        vit_path = './pretrained_models/vit_base_patch16_224'
        self.backbone = ViTModel.from_pretrained(vit_path, local_files_only=True)
        
        # Classification head
        self.fc = nn.Linear(768, num_hoi_classes)
        
    def forward(self, x):
        # Channel reduction: 4 -> 3 channels
        x = self.pre_conv(x)
        
        # ViT forward pass
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        # Classification
        out = F.sigmoid(self.fc(features))
        return out
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
      
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.ConvTranspose2d(3, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
        )
        
        
    def forward(self, x):
        en = self.encoder(x)
        out = self.decoder(en)
  
        return out