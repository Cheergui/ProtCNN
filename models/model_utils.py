import torch
import torch.nn.functional as F

class Lambda(torch.nn.Module):
    """
    A wrapper module that applies a given function as a layer in a neural network.

    This module allows for the use of arbitrary functions as layers in PyTorch models. It is particularly useful when custom operations are needed within a model but don't naturally fit into the standard layer types provided by PyTorch.

    Parameters
    ----------
    func : function
        The function to be applied to the input.

    Attributes
    ----------
    func : function
        The function that will be applied in the forward pass.
    """
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        """
        Apply the stored function to the input.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to which the function `func` will be applied.

        Returns
        -------
        torch.Tensor
            The output tensor resulting from applying the function `func` to `x`.
        """
        return self.func(x)

class ResidualBlock(torch.nn.Module):
    """
    A residual block module used in ProtCNN, designed for processing sequential data.

    This module implements a specific kind of residual block particularly suited for sequence data like protein sequences. It consists of two main convolutional layers with batch normalization and ReLU activations, and a skip connection that adds the input directly to the output of these layers.

    Parameters
    ----------
    in_channels : int
        The number of channels (feature maps) of the input.
    out_channels : int
        The number of channels (feature maps) after the first convolution.
    kernel_size : int
        The size of the convolutional kernel.
    bias : bool
        Whether to add a learnable bias to the output.
    dilation : int
        The dilation rate of the kernel in the first convolutional layer.
    padding : int
        The amount of padding applied to the input on both sides.

    Attributes
    ----------
    skip : torch.nn.Sequential
        A skip connection that adds the input directly to the output.
    bn1 : torch.nn.BatchNorm1d
        Batch normalization applied to the input of the first convolutional layer.
    conv1 : torch.nn.Conv1d
        The first convolutional layer.
    bn2 : torch.nn.BatchNorm1d
        Batch normalization applied to the input of the second convolutional layer.
    conv2 : torch.nn.Conv1d
        The second convolutional layer.
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
        """
        Apply the operations of the residual block to the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the residual block.

        Returns
        -------
        torch.Tensor
            The output tensor from the residual block.
        """
        # Execute the required layers and functions
        activation = F.relu(self.bn1(x))
        x1 = self.conv1(activation)
        x2 = self.conv2(F.relu(self.bn2(x1)))

        return x2 + self.skip(x)