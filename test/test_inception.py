from dnn_split.model_util import *
import torchvision.models as models

if __name__ == '__main__':
    inception_v3 = models.inception_v3(pretrained=False)
    inception_v3.load_state_dict(torch.load("../models/inception_v3_google-1a9a5a14.pth"))
    inception_layers = get_all_layers(inception_v3)

    for i in inception_layers:
        print(i)
        print("--------------------")