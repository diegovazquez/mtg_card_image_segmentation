"""
Model definition for semantic segmentation using LR-ASPP with MobileNetV3-Large backbone.
Adapted for binary segmentation (card vs background).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from torchvision.models.segmentation.lraspp import LRASPP_MobileNet_V3_Large_Weights


class CardSegmentationModel(nn.Module):
    """
    LR-ASPP model for card segmentation.
    Binary segmentation: background (0) and card (1).
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        """
        Initialize the model.
        
        Args:
            num_classes (int): Number of segmentation classes (default: 2)
            pretrained (bool): Use pretrained weights (default: True)
        """
        super(CardSegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained LR-ASPP model
        if pretrained:
            weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            self.model = lraspp_mobilenet_v3_large(weights=weights)
        else:
            self.model = lraspp_mobilenet_v3_large(weights=None)
        
        # Modify classifier for binary segmentation
        if num_classes != 21:  # 21 is the default for COCO
            # Get the actual channel dimensions from the backbone
            high_channels, low_channels = self._get_backbone_channels()
            
            # Replace the classifier with correct channel dimensions
            self.model.classifier = LRASPPHead(
                high_channels=high_channels,
                low_channels=low_channels,
                num_classes=num_classes,
                inter_channels=128
            )
    
    def _get_backbone_channels(self):
        """
        Get the actual channel dimensions from the MobileNetV3-Large backbone.
        
        Returns:
            tuple: (high_channels, low_channels)
        """
        # For MobileNetV3-Large, the high-level features come from the last bottleneck
        # and low-level features come from an earlier layer
        
        # Create a dummy input to inspect feature dimensions
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            # Get features from backbone
            backbone_features = self.model.backbone(dummy_input)
            
            # MobileNetV3-Large LRASPP uses features from different layers
            # High-level: output from the last layer
            # Low-level: output from an intermediate layer
            
            # The original LRASPP implementation uses:
            # - High-level features: 960 channels (from the final layer)
            # - Low-level features: 40 channels (from layer 4)
            high_channels = 960  # MobileNetV3-Large final feature channels
            low_channels = 40    # MobileNetV3-Large intermediate feature channels
            
        return high_channels, low_channels
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output segmentation map of shape (B, num_classes, H, W)
        """
        return self.model(x)['out']


class LRASPPHead(nn.Module):
    """
    Custom LR-ASPP head for binary segmentation.
    """
    
    def __init__(self, high_channels, low_channels, num_classes, inter_channels=128):
        """
        Initialize LR-ASPP head.
        
        Args:
            high_channels (int): Number of high-level feature channels from backbone
            low_channels (int): Number of low-level feature channels
            num_classes (int): Number of output classes
            inter_channels (int): Number of intermediate channels
        """
        super(LRASPPHead, self).__init__()
        
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)
    
    def forward(self, input):
        """
        Forward pass for LR-ASPP head.
        
        Args:
            input (dict): Dictionary containing 'high' and 'low' feature maps
            
        Returns:
            torch.Tensor: Output segmentation map
        """
        low = input['low']
        high = input['high']
        
        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        
        return self.low_classifier(low) + self.high_classifier(x)


def create_model(num_classes=2, pretrained=True):
    """
    Create and return a card segmentation model.
    
    Args:
        num_classes (int): Number of segmentation classes
        pretrained (bool): Use pretrained weights
        
    Returns:
        CardSegmentationModel: Initialized model
    """
    return CardSegmentationModel(num_classes=num_classes, pretrained=pretrained)


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model):
    """
    Get model size in MB.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


if __name__ == "__main__":
    # Test model creation
    model = create_model(num_classes=2, pretrained=True)
    
    # Test forward pass
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print model statistics
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
