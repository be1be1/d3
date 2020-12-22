from dnn_split.model_util import *
import torch.nn.functional as F
import math

MODEL_PATH = '../models/'


class ModelCanyon(nn.Module):
    def __init__(self, model, start, end):
        super(ModelCanyon, self).__init__()
        layers = get_all_layers(model)
        self.partialLayers = get_partial_layers(layers, start, end)
        self.x_trains = nn.ModuleList(self.partialLayers)

    def forward(self, x):
        for i in range(len(self.partialLayers)):
            x = self.partialLayers[i](x)
            if isinstance(self.partialLayers[i], nn.AdaptiveAvgPool2d):
                x = torch.flatten(x, 1)
        return x

class ModelCanyonG(nn.Module):
    def __init__(self, model, G):
        super(ModelCanyonG, self).__init__()
        self.layers = get_all_layers()
        print(self.layers)

    def forward(self):
        None


if __name__ == "__main__":

    # input = get_input()
    # resnet34 = get_pretrained_resnet34()
    # model_size = get_model_size(resnet34)
    # print(model_size)
    # model = ModelCanyon(model=resnet34, start=0, end=model_size-1)
    # model.eval()
    # output = model(input)

    startLayer = 0
    endLayer = 2
    pretrained_alexnet = get_pretrained_alexnet()

    path = MODEL_PATH + "partialmodel.pth"
    updatedModel = ModelCanyon(model=pretrained_alexnet, start=startLayer, end=endLayer)
    if isinstance(updatedModel.partialLayers[0], nn.Conv2d):
        print(updatedModel.partialLayers[0])
    print(updatedModel.partialLayers)
    torch.save(updatedModel, path)

    startLayer = 3
    endLayer = 20
    path2 = MODEL_PATH + "partialmodel2.pth"
    updatedModel2 = ModelCanyon(model=pretrained_alexnet, start=startLayer, end=endLayer)
    print(updatedModel2.partialLayers)
    torch.save(updatedModel2, path2)
