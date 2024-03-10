import torch
import torch.nn.functional as F

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ResidualBlock(torch.nn.Module):
    """
    The residual block used by ProtCNN (https://www.biorxiv.org/content/10.1101/626507v3.full).

    Args:
        in_channels: The number of channels (feature maps) of the incoming embedding
        out_channels: The number of channels after the first convolution
        dilation: Dilation rate of the first convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias, dilation, padding):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dilation = dilation
        self.padding = padding
        
        # Initialize the required layers
        self.skip = torch.nn.Sequential()

        self.bn1 = torch.nn.BatchNorm1d(num_features=self.in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels=self.in_channels, 
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size, 
                                     bias=self.bias, 
                                     dilation=self.dilation, 
                                     padding=self.padding)
        
        self.bn2 = torch.nn.BatchNorm1d(num_features=self.out_channels)
        self.conv2 = torch.nn.Conv1d(in_channels=self.out_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size,
                                     bias=self.bias,
                                     padding=self.padding)

    def forward(self, x):
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))

        return x2 + self.skip(x)