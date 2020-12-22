from dnn_split.model_util import *
import torchvision.models as models

if __name__ == '__main__':
    resnet34 = models.resnet34(pretrained=False)
    resnet34.load_state_dict(torch.load("../models/resnet34-333f7ec4.pth"))
    resnet_layers = get_all_layers(resnet34)

    for i in resnet_layers:
        print(i,"##")

    alexnet = models.alexnet(pretrained=False)
    alexnet.load_state_dict(torch.load("../models/alexnet-owt-4df8aa71.pth"))
    alexnet_layers = get_all_layers(alexnet)

    for j in alexnet_layers:
        print(j)

    alexnet = models.alexnet(pretrained=False)
    alexnet.load_state_dict(torch.load("../models/alexnet-owt-4df8aa71.pth"))
    for alexnet_module in alexnet.modules():
        if (type(alexnet_module) == nn.Sequential):
            print("type is: ", type(alexnet_module))
        print(alexnet_module)


    def get_all_layers(model):
        # submodel = nn.Sequential(*list(model.children()))
        layers = [module for module in model.modules() if type(module) != nn.Sequential]
        return layers[1:]

