"""
vision_teacher_network.py
=============================================================================
Cross-Modal Vision-Based Teacher Network (OpenPose / ResNet3D Architecture):
- Ingests synchronized RGB optical camera streams [Batch, 3, Frames, Height, Width]
- Extracts 3D Spatio-Temporal Human Pose & Dense Body Feature Representations
- Supervises the RFStudentNetwork via Cross-Modal Knowledge Distillation (KD)
- Operates during calibration and multi-modal training sessions
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class VisionTeacher3D(nn.Module):
    """
    3D ResNet-based Spatio-Temporal Vision Teacher Network.
    Ingests synchronized RGB video frames (B, 3, T, H, W) and generates:
    1. Spatial Heatmaps & 3D Keypoint Coordinates [B, 17, 3]
    2. Latent Visual Feature Embeddings [B, 512, S_lat, T_lat] for RF KD alignment.
    """
    def __init__(self, num_joints: int = 17, pretrained_weights: bool = False):
        super().__init__()
        self.num_joints = num_joints

        # Spatio-Temporal 3D Conv Encoder
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.layer1 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=(1, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True)
        )

        # Global Spatio-Temporal Pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 3D Joint Regression Head
        self.keypoint_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_joints * 3)
        )

    def forward(self, rgb_frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass.
        Args:
            rgb_frames: [Batch, Channels=3, Time, Height, Width] (e.g., [B, 3, 16, 224, 224])
        Returns:
            Dictionary containing 3D keypoints and latent teacher feature maps.
        """
        b = rgb_frames.shape[0]

        f1 = self.stem(rgb_frames)    # [B, 64, T, H/4, W/4]
        f2 = self.layer1(f1)          # [B, 128, T, H/8, W/8]
        f3 = self.layer2(f2)          # [B, 256, T/2, H/16, W/16]
        f4 = self.layer3(f3)          # [B, 512, T/4, H/32, W/32]

        pooled = self.global_pool(f4).view(b, 512)
        pose_3d = self.keypoint_head(pooled).view(b, self.num_joints, 3)

        # Project 3D feature map to 2D for cross-modal alignment with student RF latent map
        # Collapse spatial dimensions: [B, 512, T/4, H/32, W/32] -> [B, 512, H_eff, T_eff]
        teacher_kd_map = torch.mean(f4, dim=4)  # [B, 512, T/4, H/32]

        return {
            "teacher_pose_3d": pose_3d,
            "teacher_kd_feature_map": teacher_kd_map,
            "global_visual_embedding": pooled
        }


class CrossModalDistillationLoss(nn.Module):
    """
    Supervises the RF Student Network using Teacher's Vision Representations:
    L_distill = L_pose_teacher + alpha * L_feature_alignment
    """
    def __init__(self, alpha_feat: float = 0.5):
        super().__init__()
        self.alpha_feat = alpha_feat
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        student_predictions: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # 1. Pose mimicry loss
        loss_pose_distill = self.smooth_l1(
            student_predictions["pose_3d"],
            teacher_outputs["teacher_pose_3d"]
        )

        # 2. Feature map alignment (Cosine + MSE)
        s_feat = F.adaptive_avg_pool2d(student_predictions["latent_feature_map"], (1, 1)).flatten(1)
        t_feat = teacher_outputs["global_visual_embedding"]
        
        loss_feat = (
            1.0 - F.cosine_similarity(s_feat, t_feat).mean() +
            0.1 * self.mse(s_feat, t_feat)
        )

        return loss_pose_distill + self.alpha_feat * loss_feat
