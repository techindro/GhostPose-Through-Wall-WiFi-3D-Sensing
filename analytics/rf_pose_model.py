"""
rf_pose_model.py
=============================================================================
PyTorch Implementation of RFStudentNetwork & Multi-Task Loss:
- Spatio-Temporal 3D-ResNet / Temporal Convolutional Blocks (TCN)
- Spatial Attention Joint Regressor (17 Keypoints in 3D: [x, y, z])
- Identity Re-ID Metric Learning Head (Cosine/ArcFace Embeddings)
- Micro-Doppler Vital Signs Multi-Scale Head
- Kinematic Limb-Length Geometric Consistency Loss Function
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional


class SpatialTemporalConvBlock(nn.Module):
    """
    Factorized (2D+1D) Spatio-Temporal Convolution Block to extract
    spatial subcarrier features and temporal Doppler evolution.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: Tuple[int, int] = (1, 1)):
        super().__init__()
        # Spatial convolution over Subcarriers
        self.conv_spatial = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1),
            stride=stride, padding=(1, 0), bias=False
        )
        self.bn_spatial = nn.BatchNorm2d(out_channels)
        
        # Temporal convolution over Time
        self.conv_temporal = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3),
            stride=(1, 1), padding=(0, 1), bias=False
        )
        self.bn_temporal = nn.BatchNorm2d(out_channels)
        
        self.act = nn.SiLU(inplace=True)
        
        # Residual projection
        if in_channels != out_channels or stride != (1, 1):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Channels, Subcarriers, Time]
        res = self.residual(x)
        out = self.act(self.bn_spatial(self.conv_spatial(x)))
        out = self.bn_temporal(self.conv_temporal(out))
        return self.act(out + res)


class TemporalSelfAttention(nn.Module):
    """Multi-Head Self-Attention over Temporal Axis for Long-Range Trajectory Modeling."""
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Time, FeatureDim]
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)


class RFStudentNetwork(nn.Module):
    """
    Through-the-Wall 3D Skeleton, Re-ID, and Vital Signs Deep Neural Network.
    """
    def __init__(
        self,
        in_channels: int = 12,        # E.g., 3 Antennas * 4 Features (Amp, Phase, I, Q)
        num_joints: int = 17,         # Standard COCO 17 3D joints
        reid_dim: int = 128,
        num_identities: int = 10,
        temporal_window: int = 100
    ):
        super().__init__()
        self.num_joints = num_joints
        self.temporal_window = temporal_window

        # 1. Front-End Spatio-Temporal RF Feature Encoder
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True)
        )

        # 2. Residual Hierarchical Backbone
        self.layer1 = SpatialTemporalConvBlock(64, 128, stride=(2, 2))
        self.layer2 = SpatialTemporalConvBlock(128, 256, stride=(2, 2))
        self.layer3 = SpatialTemporalConvBlock(256, 512, stride=(2, 1))

        # Temporal Sequence Processing
        # Target reduced temporal length
        self.reduced_time = max(temporal_window // 4, 4)
        self.temporal_pool = nn.AdaptiveAvgPool2d((1, self.reduced_time))
        self.temporal_attn = TemporalSelfAttention(feature_dim=512)

        # 3. Multi-Task Heads
        
        # --- Head A: 3D Skeleton Keypoints Regressor ---
        # Outputs [Batch, 17, 3] representing (x, y, z) metric coordinates
        self.pose_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_joints * 3)
        )

        # --- Head B: Person Identification & Re-ID Metric Head ---
        self.reid_embed_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Linear(256, reid_dim)
        )
        self.reid_classifier = nn.Linear(reid_dim, num_identities)

        # --- Head C: Contactless Vital Signs Regressor ---
        self.vital_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 2)  # [Respiration Rate (BrPM), Heart Rate (BPM)]
        )
        
        # Respiration Waveform Deconvolution Decoder
        self.waveform_fc = nn.Linear(512, 128)
        self.waveform_decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(temporal_window)
        )

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward Pass.
        Args:
            x: Input tensor [Batch, Channels, Subcarriers, Time]
        Returns:
            Dictionary containing 3D pose, Re-ID features, classification logits, and vitals.
        """
        b = x.shape[0]
        
        # Backbone Feature Extraction
        feat = self.stem(x)          # [B, 64, S/2, T/2]
        feat = self.layer1(feat)      # [B, 128, S/4, T/4]
        feat = self.layer2(feat)      # [B, 256, S/8, T/8]
        feat = self.layer3(feat)      # [B, 512, S/16, T/8]

        # Temporal Global Latent Representation
        # Squeeze subcarrier dimension: [B, 512, 1, T_seq] -> [B, T_seq, 512]
        seq_feat = self.temporal_pool(feat).squeeze(2).permute(0, 2, 1)
        seq_feat = self.temporal_attn(seq_feat)  # [B, T_seq, 512]

        # Global Pooled Embedding for Dense Classification & Regression
        global_embed = torch.mean(seq_feat, dim=1)  # [B, 512]

        # 1. 3D Pose Keypoints [B, 17, 3]
        pose_raw = self.pose_head(global_embed)
        pose_3d = pose_raw.view(b, self.num_joints, 3)

        # 2. Re-ID Metric Learning (Normalized L2 Embeddings)
        reid_feat = self.reid_embed_head(global_embed)
        reid_norm = F.normalize(reid_feat, p=2, dim=1)
        reid_logits = self.reid_classifier(reid_norm)

        # 3. Vital Signs Extraction
        vitals = self.vital_head(global_embed)  # [B, 2] -> [BrPM, BPM]
        
        # Respiration Waveform: [B, T_seq, 512] -> [B, 128, T_seq] -> [B, temporal_window]
        wf_proj = self.waveform_fc(seq_feat).permute(0, 2, 1)
        resp_waveform = self.waveform_decoder(wf_proj).squeeze(1)

        return {
            "pose_3d": pose_3d,
            "reid_features": reid_norm,
            "reid_logits": reid_logits,
            "vital_rates": vitals,
            "respiration_waveform": resp_waveform,
            "latent_feature_map": feat  # Exposed for Cross-Modal Teacher-Student KD Loss
        }


# =============================================================================
# Multi-Task Loss with Kinematic Geometric Consistency
# =============================================================================

# Standard Human Skeletal Joint Kinematic Tree (COCO 17 Topology)
COCO_BONES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),             # Head (Nose -> Eyes -> Ears)
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # Upper Body (Shoulders -> Elbows -> Wrists)
    (5, 11), (6, 12), (11, 12),                 # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)      # Lower Body (Hips -> Knees -> Ankles)
]


class RFMultiTaskLoss(nn.Module):
    """
    Composite Multi-Task Loss balancing:
    1. 3D Keypoint MPJPE / Smooth-L1 Loss
    2. Kinematic Bone-Length Invariance Constraint Loss
    3. Re-ID Cross-Entropy & Metric Cosine Separation
    4. Vital Signs Respiration & Heart Rate MSE + Spectral Loss
    5. Vision-to-RF Knowledge Distillation (Teacher-Student KD)
    """
    def __init__(
        self,
        w_pose: float = 1.0,
        w_bone: float = 0.5,
        w_reid: float = 0.3,
        w_vitals: float = 0.2,
        w_distill: float = 0.4
    ):
        super().__init__()
        self.w_pose = w_pose
        self.w_bone = w_bone
        self.w_reid = w_reid
        self.w_vitals = w_vitals
        self.w_distill = w_distill
        
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.01)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def bone_length_loss(self, pred_pose: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
        """
        Penalizes anatomical bone deformation between kinematic joints.
        pred_pose, gt_pose: [Batch, 17, 3]
        """
        loss = torch.tensor(0.0, device=pred_pose.device)
        for (j1, j2) in COCO_BONES:
            pred_bone_len = torch.norm(pred_pose[:, j1, :] - pred_pose[:, j2, :], p=2, dim=-1)
            gt_bone_len = torch.norm(gt_pose[:, j1, :] - gt_pose[:, j2, :], p=2, dim=-1)
            loss += self.smooth_l1(pred_bone_len, gt_bone_len)
        return loss / len(COCO_BONES)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        teacher_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes composite scalar loss.
        """
        pred_pose = predictions["pose_3d"]
        gt_pose = targets["gt_pose_3d"]
        
        # 1. 3D Keypoint Pose Loss
        loss_pose = self.smooth_l1(pred_pose, gt_pose)
        
        # 2. Kinematic Consistency Loss
        loss_bone = self.bone_length_loss(pred_pose, gt_pose)
        
        # 3. Person Identification Cross-Entropy Loss
        loss_reid = self.ce_loss(predictions["reid_logits"], targets["gt_identity"])
        
        # 4. Vital Signs Rate & Waveform Loss
        loss_vital_rate = self.mse_loss(predictions["vital_rates"], targets["gt_vital_rates"])
        loss_waveform = self.smooth_l1(predictions["respiration_waveform"], targets["gt_respiration_waveform"])
        loss_vitals = loss_vital_rate + 0.5 * loss_waveform
        
        # 5. Cross-Modal Teacher Distillation Loss (Cosine/MSE alignment with Vision Teacher)
        loss_kd = torch.tensor(0.0, device=pred_pose.device)
        if teacher_features is not None:
            student_feat = predictions["latent_feature_map"]
            s_pool = F.adaptive_avg_pool2d(student_feat, (1, 1)).flatten(1)
            t_pool = F.adaptive_avg_pool2d(teacher_features, (1, 1)).flatten(1)
            loss_kd = 1.0 - F.cosine_similarity(s_pool, t_pool).mean()

        total_loss = (
            self.w_pose * loss_pose +
            self.w_bone * loss_bone +
            self.w_reid * loss_reid +
            self.w_vitals * loss_vitals +
            self.w_distill * loss_kd
        )

        return {
            "total_loss": total_loss,
            "loss_pose": loss_pose,
            "loss_bone": loss_bone,
            "loss_reid": loss_reid,
            "loss_vitals": loss_vitals,
            "loss_kd": loss_kd
        }
