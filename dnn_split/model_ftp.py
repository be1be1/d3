from dnn_split.model_util import *
from dnn_split.ftp_util import *
import torch.nn.functional as F
import math

MODEL_PATH = '../models/'

class ModelFTP(nn.Module):
    """
    calculate the inference result of the partitioned tile according to its coordinates and partial padding
    """
    def __init__(self, model, start, end, coordinate, input_w, input_h):
        """
        Args:
            model: the DNN model
            start: starting number of the layer that FTP begins from
            end: ending number of the layer that FTP ends with
            coordinate: coordinates of the partitioned tile of the start layer
            input_w: width of the input feature map of the starting layer
            input_h: height of the input feature map of the starting layer
        """
        super(ModelFTP, self).__init__()
        layers = get_all_layers(model)
        self.partialLayers = get_partial_layers(layers, start, end)
        self.x_trains = nn.ModuleList(self.partialLayers)
        self.x1 = coordinate.top_left_x
        self.y1 = coordinate.top_left_y
        self.x2 = coordinate.bottom_right_x
        self.y2 = coordinate.bottom_right_y
        self.input_w = input_w
        self.input_h = input_h

    def cal(self, x, padding):
        """
        1. update the top left coordinate of the partitioned tile according to padding info
        2. padding the partitioned tile according to its top left coordinate
        """
        if len(padding) == 2:         # conv layer
            pad0 = padding[0]
            pad1 = padding[1]
        else:                         # pooling layer
            pad0 = padding
            pad1 = padding

        if self.x1 == 0:
            self.x1 = 0
            x = F.pad(input=x, pad=[pad0, 0, 0, 0], mode='constant', value=0)
        else:
            self.x1 = self.x1 + pad0

        if self.y1 == 0:
            self.y1 = 0
            x = F.pad(input=x, pad=[0, 0, pad1, 0], mode='constant', value=0)
        else:
            self.y1 = self.y1 + pad1
        return x, self.x1, self.y1

    def forward(self, x):
        for i in range(len(self.partialLayers)):
            if isinstance(self.partialLayers[i], nn.Conv2d):       # conv layer
                in_ch = self.partialLayers[i].in_channels
                out_ch = self.partialLayers[i].out_channels
                kernel = self.partialLayers[i].kernel_size
                stride = self.partialLayers[i].stride
                padding = self.partialLayers[i].padding
                weight = self.partialLayers[i].weight
                bias = self.partialLayers[i].bias

                if padding[0] != 0:
                    if self.x2 == self.input_w and self.y2 == self.input_h:     # the bottom right coordinates locates at the bottom right corner
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + 2 * padding[0]
                        self.y2 = self.y2 + 2 * padding[1]
                        x = F.pad(input=x, pad=[0, padding[0], 0, padding[1]], mode='constant', value=0)   # padding the partitioned tile according to its right bottom coordinate
                    elif self.x2 == self.input_w:    # the bottom right coordinates locates at the right side of the feature map
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + 2 * padding[0]
                        self.y2 = self.y2 + padding[1]
                        x = F.pad(input=x, pad=[0, padding[0], 0, 0], mode='constant', value=0)   # padding the partitioned tile according to its right bottom coordinate
                    elif self.y2 == self.input_h:      # the bottom right coordinates locates at the down side of the feature map
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + padding[0]
                        self.y2 = self.y2 + 2 * padding[1]
                        x = F.pad(input=x, pad=[0, 0, 0, padding[1]], mode='constant', value=0)   # padding the partitioned tile according to its right bottom coordinate
                    else:                # the bottom right coordinates locates at the middle of the feature map
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + padding[0]
                        self.y2 = self.y2 + padding[1]

                # calculate the coordinate of next layer
                self.x1 = math.floor(self.x1 / stride[0])
                self.y1 = math.floor(self.y1 / stride[1])

                self.x2 = math.floor((self.x2 - kernel[0]) / stride[0] + 1)
                self.y2 = math.floor((self.y2 - kernel[1]) / stride[1] + 1)

                # calculate the width and height of the input feature map of next layer
                self.input_h = math.floor((self.input_h - kernel[1] + 2 * padding[1]) / stride[1] + 1)
                self.input_w = math.floor((self.input_w - kernel[0] + 2 * padding[0]) / stride[0] + 1)

                # change padding of the current layer to 0
                self.partialLayers[i] = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel[0],
                                                  stride=stride[0], padding=0)
                with torch.no_grad():
                    self.partialLayers[i].weight = nn.Parameter(weight)
                    self.partialLayers[i].bias = nn.Parameter(bias)

            if isinstance(self.partialLayers[i], nn.MaxPool2d):        # pooling layer
                kernel = self.partialLayers[i].kernel_size
                stride = self.partialLayers[i].stride
                padding = self.partialLayers[i].padding

                if padding != 0:
                    if self.x2 == self.input_w and self.y2 == self.input_h:   # the bottom right coordinates locates at the bottom right corner
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + 2 * padding
                        self.y2 = self.y2 + 2 * padding
                        x = F.pad(input=x, pad=[0, padding, 0, padding], mode='constant', value=0)   # padding the partitioned tile according to its right bottom coordinate
                    elif self.x2 == self.input_w:      # the bottom right coordinates locates at the right side of the feature map
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + 2 * padding
                        self.y2 = self.y2 + padding
                        x = F.pad(input=x, pad=[0, padding, 0, 0], mode='constant', value=0)    # padding the partitioned tile according to its right bottom coordinate
                    elif self.y2 == self.input_h:    # the bottom right coordinates locates at the down side of the feature map
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + padding
                        self.y2 = self.y2 + 2 * padding
                        x = F.pad(input=x, pad=[0, 0, 0, padding], mode='constant', value=0)    # padding the partitioned tile according to its right bottom coordinate
                    else:             # the bottom right coordinates locates at the middle of the feature map
                        x, self.x1, self.y1 = self.cal(x, padding)

                        self.x2 = self.x2 + padding
                        self.y2 = self.y2 + padding

                # calculate the coordinate of next layer
                self.x1 = math.floor(self.x1 / stride)
                self.y1 = math.floor(self.y1 / stride)

                self.x2 = math.floor((self.x2 - kernel) / stride + 1)
                self.y2 = math.floor((self.y2 - kernel) / stride + 1)

                # calculate the width and height of the input feature map of next layer
                self.input_h = math.floor((self.input_h - kernel + 2 * padding) / stride + 1)
                self.input_w = math.floor((self.input_w - kernel + 2 * padding) / stride + 1)

                # change padding of the current layer to 0
                self.partialLayers[i] = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=0)

            print("The x after adding padding is", x.size())
            x = self.partialLayers[i](x)
            print("The x after processing is", x.size())
            if isinstance(self.partialLayers[i], nn.AdaptiveAvgPool2d):
                x = torch.flatten(x, 1)

        print("----------------------------------------------------------------")
        return x