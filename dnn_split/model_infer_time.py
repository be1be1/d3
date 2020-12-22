from dnn_split.model_util import *
import time
import torch
import torch.nn as nn

PATH = "../model/alexnet-owt-4df8aa71.pth"

class ModelInferTime(nn.Module):

    def __init__(self, model, start, end):
        super(ModelInferTime, self).__init__()
        layers = get_all_layers(model)
        self.partialLayers = get_partial_layers(layers, start, end)
        self.x_train = nn.ModuleList(self.partialLayers)

    def get_mul(self, arr):
        mul = 1
        for i in arr:
            mul = mul * i
        return mul

    def forward(self, x):
        infer_time = []
        x_size = []
        input_size = self.get_mul(x.size()[1:])
        for layer in self.partialLayers:
            start_time = time.time()
            x = layer(x)
            end_time = time.time()
            running_time = end_time - start_time
            x_size.append(self.get_mul(x.size()[1:]))
            # print("time cost of No.", i, "layer is: %.*f sec" %(9, running_time))
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                start_time = time.time()
                x = torch.flatten(x, 1)
                end_time = time.time()
                flatten_running_time = end_time - start_time
                running_time = running_time + flatten_running_time

            infer_time.append(running_time)

        x_size[-1] = input_size
        return x_size, infer_time


class ModelInferTimeGPU(nn.Module):

    def __init__(self, model, start, end):
        super(ModelInferTimeGPU, self).__init__()
        layers = get_all_layers(model)
        self.partialLayers = get_partial_layers(layers, start, end)
        self.x_train = nn.ModuleList(self.partialLayers)

    def get_mul(self, arr):
        mul = 1
        for i in arr:
            mul = mul * i
        return mul

    def forward(self, x):
        infer_time = []
        x_size = []
        input_size = self.get_mul(x.size()[1:])
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        for layer in self.partialLayers:
            start_time.record()
            x = layer(x)
            end_time.record()
            torch.cuda.synchronize()
            running_time = start_time.elapsed_time(end_time)/1000
            x_size.append(self.get_mul(x.size()[1:]))
            # print("time cost of No.", i, "layer is: %.*f sec" %(9, running_time))
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                start_time.record()
                x = torch.flatten(x, 1)
                end_time.record()
                torch.cuda.synchronize()
                flatten_running_time = start_time.elapsed_time(end_time)/1000
                running_time = running_time + flatten_running_time

            infer_time.append(running_time)

        x_size[-1] = input_size
        return x_size, infer_time


if __name__ == "__main__":

    input = get_input()

    resnet34 = get_pretrained_resnet34()
    model_size = get_model_size(resnet34)
    print(model_size)
    model = ModelInferTime(model=resnet34, start=0, end=model_size-1)
    model.eval()
    x_size, output = model(input)
    print(len(x_size))