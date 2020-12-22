from dnn_split.model_util import *
import torchvision.models as models



if __name__ == '__main__':
    alexnet = models.alexnet(pretrained=False)
    alexnet.load_state_dict(torch.load("../models/alexnet-owt-4df8aa71.pth"))
    alexnet_layers = get_all_layers(alexnet)

    for i in alexnet_layers:
        print(i)
        print("---------------")