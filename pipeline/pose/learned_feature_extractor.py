# pipeline/pose/learned_feature_extractor.py
"""
Learned Feature Extractor using CNN/Transformer
Replaces handcrafted features with learned representations for better accuracy
"""
import numpy as np
import logging
import os

log = logging.getLogger("learned_features")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available; learned features disabled")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    log.warning("PyTorch not available; learned features disabled")


class TemporalCNN(nn.Module):
    """
    Temporal CNN for extracting learned features from pose sequences.
    Uses 1D convolutions to capture temporal patterns.
    """
    
    def __init__(self, input_dim=51,  # 17 keypoints × 3 (x, y, confidence)
                 feature_dim=256, num_layers=3):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        
        # 1D Convolutional layers
        layers = []
        in_channels = input_dim
        out_channels = 128
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_channels = out_channels
            if i < num_layers - 1:
                out_channels = min(out_channels * 2, 256)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(in_channels, feature_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Transpose for Conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = self.conv_layers(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, channels)
        
        # Projection
        x = self.projection(x)
        
        return x


class PoseTransformerEncoder(nn.Module):
    """
    Transformer encoder for pose sequence feature extraction.
    Uses self-attention to capture long-range temporal dependencies.
    """
    
    def __init__(self, input_dim=51, d_model=256, nhead=8, 
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learned)
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))  # Max 1000 frames
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, 256)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        
        # Project input
        x = self.input_proj(x) * np.sqrt(self.d_model)
        
        # Add positional encoding
        pos = self.pos_encoder[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)  # (batch, d_model)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class LearnedFeatureExtractor:
    """
    Learned feature extractor that replaces handcrafted features.
    Can use either TemporalCNN or Transformer encoder.
    """
    
    def __init__(self, method="transformer",  # "cnn" or "transformer"
                 model_path=None, device="cpu", 
                 input_dim=51,  # 17 keypoints × 3
                 feature_dim=256):
        self.method = method
        self.device = device
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.model = None
        
        if not TORCH_AVAILABLE:
            log.warning("PyTorch not available; learned features disabled")
            return
        
        # Initialize model
        if method == "cnn":
            self.model = TemporalCNN(input_dim=input_dim, feature_dim=feature_dim)
        elif method == "transformer":
            self.model = PoseTransformerEncoder(
                input_dim=input_dim, 
                d_model=256,
                feature_dim=feature_dim
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.model.to(device)
        self.model.eval()
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                log.info("Loaded learned feature extractor from: %s", model_path)
            except Exception as e:
                log.warning("Failed to load learned feature model: %s", e)
        else:
            log.info("Initialized new learned feature extractor (method: %s)", method)
    
    def extract_features(self, kps_sequence):
        """
        Extract learned features from keypoint sequence.
        
        Args:
            kps_sequence: List of keypoint lists, each with 17 keypoints (x, y, confidence)
                         Shape: (seq_len, 17, 3) or list of lists
        
        Returns:
            Feature vector of shape (feature_dim,)
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback to handcrafted features
            log.debug("Learned features not available, using fallback")
            return None
        
        try:
            # Convert to tensor format
            if isinstance(kps_sequence, list):
                # Flatten: (seq_len, 17, 3) -> (seq_len, 51)
                kps_array = []
                for kps in kps_sequence:
                    if isinstance(kps, list):
                        flat = []
                        for kp in kps:
                            if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                                flat.extend([kp[0], kp[1], kp[2]])
                            else:
                                flat.extend([0.0, 0.0, 0.0])
                        kps_array.append(flat)
                    else:
                        kps_array.append([0.0] * 51)
                kps_tensor = torch.FloatTensor(kps_array).unsqueeze(0)  # (1, seq_len, 51)
            else:
                # Assume numpy array
                if kps_sequence.ndim == 3:
                    # (seq_len, 17, 3) -> (seq_len, 51)
                    kps_tensor = torch.FloatTensor(
                        kps_sequence.reshape(kps_sequence.shape[0], -1)
                    ).unsqueeze(0)
                else:
                    kps_tensor = torch.FloatTensor(kps_sequence).unsqueeze(0)
            
            kps_tensor = kps_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(kps_tensor)
                features = features.squeeze(0).cpu().numpy()  # (feature_dim,)
            
            return features
            
        except Exception as e:
            log.exception("Error extracting learned features: %s", e)
            return None
    
    def save_model(self, save_path):
        """Save model for inference."""
        if self.model is not None:
            torch.save(self.model.state_dict(), save_path)
            log.info("Saved learned feature extractor to: %s", save_path)


# Hybrid feature extractor: combines learned + handcrafted
class HybridFeatureExtractor:
    """
    Combines learned features with handcrafted features for best of both worlds.
    """
    
    def __init__(self, learned_extractor, handcrafted_extractor):
        self.learned_extractor = learned_extractor
        self.handcrafted_extractor = handcrafted_extractor
    
    def extract_features(self, kps_sequence, prev_kps=None, prev_prev_kps=None):
        """
        Extract hybrid features (learned + handcrafted).
        
        Returns:
            Combined feature vector
        """
        features = []
        
        # Learned features
        if self.learned_extractor:
            learned_feat = self.learned_extractor.extract_features(kps_sequence)
            if learned_feat is not None:
                features.extend(learned_feat.tolist())
        
        # Handcrafted features (from current frame)
        if self.handcrafted_extractor and len(kps_sequence) > 0:
            current_kps = kps_sequence[-1] if isinstance(kps_sequence, list) else kps_sequence[-1]
            handcrafted_feat = self.handcrafted_extractor.extract_feature_vector(
                current_kps, prev_kps, prev_prev_kps
            )
            if handcrafted_feat is not None:
                features.extend(handcrafted_feat.tolist())
        
        return np.array(features, dtype=np.float32) if features else None

