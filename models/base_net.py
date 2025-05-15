import torch
import torch.nn as nn

class BaseNet(nn.Module):
    """Base network architecture for JULE."""
    
    def __init__(self, input_channels=1, feature_dim=10):
        super(BaseNet, self).__init__()
        
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv2
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Placeholder for classifier layers
        self.classifier = None
        self.feature_dim = feature_dim
        
    def _create_classifier(self, x):
        """
        Create classifier layers based on the output shape of convolutional layers.
        Args:
            x: Input tensor to get shape and device information
        """
        flattened_dim = x.shape[1] * x.shape[2] * x.shape[3]
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.feature_dim)
        )
        # Move classifier to the same device as input
        self.classifier = self.classifier.to(x.device)
        
    def forward(self, x):
        x = self.features(x)
        
        # Create classifier on first forward pass when we know the input size
        if self.classifier is None:
            self._create_classifier(x)
            
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_embedding(self, x):
        """Get the embedding of the input."""
        return self.forward(x) 