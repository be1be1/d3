import torch
from dnn_split.model_util import get_alexnet
from dnn_split.model_canyon import ModelCanyon
from PIL import Image
from torchvision import transforms


MODEL_PATH = './models/'
IMAGE_PATH = './data/images/'


def get_input():
    input_image = Image.open(IMAGE_PATH+'dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch

if __name__ == '__main__':
    input = get_input()
    path = MODEL_PATH+"partialmodel.pth"
    alexnet = get_alexnet()
    model = ModelCanyon(model=alexnet, start=0, end=2)
    model = torch.load(path)
    model.eval()
    # print(model.partialLayers)
    output = model(input)

    path2 = MODEL_PATH+"partialmodel2.pth"
    model2 = ModelCanyon(model=alexnet, start=3, end=20)
    model2 = torch.load(path2)
    model2.eval()
    output2 = model2(output)
    print(output2)