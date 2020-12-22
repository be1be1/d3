import torch
from dnn_split.model_canyon import ModelCanyon
from dnn_split.model_util import get_alexnet, get_pretrained_alexnet
from PIL import Image
from torchvision import transforms

MODEL_PATH = '../data/models/'
IMAGE_PATH = '../data/images/'


def get_input():
    input_image = Image.open(IMAGE_PATH + 'dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


if __name__ == "__main__":

    # input = get_input()
    # path = "../models/partialmodel.pth"
    # alexnet = get_alexnet()
    # model = ModelCanyon(model=alexnet, start=0, end=2)
    # model = torch.load(path)
    # model.eval()
    # # print(model.partialLayers)
    # output = model(input)
    #
    # path2 = "../models/partialmodel2.pth"
    # model2 = ModelCanyon(model=alexnet, start=3, end=20)
    # model2 = torch.load(path2)
    # model2.eval()
    # print(model2.partialLayers)
    # output2 = model2(output)
    # print(output2)

    input2 = get_input()
    startLayer = 0
    endLayer = 2
    model3 = get_pretrained_alexnet()
    model3.eval()
    output3 = model3(input2)
    # print("#####################################")
    print(output3)