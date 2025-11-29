import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, return_layer='layer3'):
        super().__init__()
        # Load standard ResNet18
        # Note: 'pretrained=True' is deprecated in newer versions, use 'weights'
        # But for compatibility with older torch versions in requirements, we check.
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.net = resnet18(weights=weights)
        except ImportError:
            self.net = resnet18(pretrained=pretrained)
            
        self.return_layer = return_layer
        
        # We don't need the FC layer
        del self.net.fc
        del self.net.avgpool

    def forward(self, x):
        """
        x: (B, 3, H, W)
        """
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x) # /4
        if self.return_layer == 'layer1': return x
        
        x = self.net.layer2(x) # /8
        if self.return_layer == 'layer2': return x
        
        x = self.net.layer3(x) # /16 (14x14 for 224)
        if self.return_layer == 'layer3': return x
        
        x = self.net.layer4(x) # /32 (7x7 for 224)
        return x
