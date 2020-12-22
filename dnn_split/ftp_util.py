from dnn_split.model_util import *
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class TileRegion:
    """
    define the coordination of a feature map
    (top_left_x, top_left_y) represents the top left coordination
    (bottom_right_x, bottom_right_y) represents the bottom right coordination
    """
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int

@dataclass
class NetPara:
    """
    define the net para of each layer
    (stride, kernel_size, padding): filter para of each layer
    type: convolution or pooling
    input_width: width of the input feature maps of each layer
    input_height: height of the input feature maps of each layer
    """
    stride: int
    kernel_size: int
    padding: int
    type: str
    input_width: int
    input_height: int

@dataclass
class FtpPara:
    """
    define the para for FTP algorithm
    partitions_w: the number of slices divided from width
    partitions_h: the number of slices divided from height
    fused_layers: the number of layers that need to be partitioned by FTP algo
    task_id: id for each partition
    input_tiles: TileRegion info of each partition of the input feature maps
    output_tiles: TileRegion info of each partition of the output feature maps
    """
    partitions_w: int
    partitions_h: int
    fused_layers: int
    task_id: int
    input_tiles: TileRegion
    output_tiles: TileRegion

class ModelInterpreter(nn.Module):
    """ Interpret the model layer by layer
    Retrieve the parameters of each layer (convolution or pooling) for the DNN model, including:
    1. feature map size
    2. kernel size
    3. stride
    4. padding
    5. layer type: convolution or pooling
    """
    def __init__(self, model):
        super(ModelInterpreter, self).__init__()
        self.layers = get_all_layers(model)        # get each layer of the DNN model
        self.x_train = nn.ModuleList(self.layers)

    def forward(self, x):
        x_size = []
        x_kernel_size = []
        x_stride = []
        x_padding = []
        x_type = []

        for i in range(len(self.layers)):       # loop over all the layers
            # forward layer by layer
            x = self.layers[i](x)

            # add flatten after AvgPool
            if isinstance(self.layers[i], nn.AdaptiveAvgPool2d):
                x = torch.flatten(x, 1)

            # get kernel size of the current layer
            if hasattr(self.layers[i], 'kernel_size'):
                x_kernel_size.append(self.layers[i].kernel_size)
            else:
                continue

            # get stride of the current layer
            if hasattr(self.layers[i], 'stride'):
                x_stride.append(self.layers[i].stride)
            else:
                continue

            # get padding of the current layer
            if hasattr(self.layers[i], 'padding'):
                x_padding.append(self.layers[i].padding)
            else:
                continue

            # get layer type of the current layer
            if isinstance(self.layers[i], nn.Conv2d):
                x_type.append("convolution")
            elif isinstance(self.layers[i], nn.MaxPool2d) or isinstance(self.layers[i], nn.MinPool2d):
                x_type.append("pooling")
            else:
                continue

            # get output feature map size of all dimensions except the batch dimension (channel, width, height)
            x_size.append(x.size()[1:])

        return x_size, x_kernel_size, x_stride, x_padding, x_type


def load_dnn_model(input_size, layer_size, layer_kernel_size, layer_stride, layer_padding, layer_type):
    """ load dnn model and retrieve relevant parameters
    Args:
        input_size: the input image size (channel, height, width)
        layer_size: output feature map size of each layer (convolution or pooling layer)
        layer_kernel_size: kernel size para of each layer (convolution or pooling layer)
        layer_stride: stride para of each layer (convolution or pooling layer)
        layer_padding: padding para of each layer (convolution or pooling layer)
        layer_type: type of each layer (convolution or pooling layer)

    Returns:
        net_para: necessary parameters of each layer for subsequent FTP calculation, including:
        1. stride
        2. kernel size
        3. padding
        4. type
        5. input_width
        6. output_width
    """
    net_para = [[0] for _ in range(len(layer_size))]
    input_width = [[0] for _ in range(len(layer_size))]
    input_height = [[0] for _ in range(len(layer_size))]
    for i in range(len(layer_size)):
        # assign the relevant para of the input maps for each layer
        if i == 0:
            input_width[i] = input_size[2]
            input_height[i] = input_size[1]
        else:
            input_width[i] = layer_size[i-1][2]
            input_height[i] = layer_size[i-1][1]

        # calculate the net_para
        if np.array(layer_stride[i]).size == 2:
            net_para[i] = NetPara(layer_stride[i][0], layer_kernel_size[i][0], layer_padding[i][0], layer_type[i], input_width[i], input_height[i])
        else:
            net_para[i] = NetPara(layer_stride[i], layer_kernel_size[i], layer_padding[i], layer_type[i], input_width[i], input_height[i])

    return net_para
