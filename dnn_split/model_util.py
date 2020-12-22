import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

MODEL_PATH = "../models/"
IMAGE_PATH = '../data/images/'


def get_alexnet():
    alexnet = models.alexnet(pretrained=False)
    return alexnet


def get_pretrained_alexnet():
    pretrained_alexnet = models.alexnet(pretrained=False)
    pretrained_alexnet.load_state_dict(torch.load(MODEL_PATH + 'alexnet-owt-4df8aa71.pth'))

    return pretrained_alexnet


def get_pretrained_vgg16():
    pretrained_vgg16 = models.vgg16(pretrained=False)
    pretrained_vgg16.load_state_dict(torch.load(MODEL_PATH + 'alexnet-owt-4df8aa71.pth'))

    return pretrained_vgg16



def get_resnet34():
    resnet34 = models.resnet34(pretrained=False)
    return resnet34


def get_pretrained_resnet34():
    pretrained_resnet34 = models.resnet34(pretrained=False)
    pretrained_resnet34.load_state_dict(torch.load(MODEL_PATH + "resnet34-333f7ec4.pth"))
    return pretrained_resnet34


def get_all_layers(model):
    # submodel = nn.Sequential(*list(model.children()))
    # layers = [module for module in model.modules() if type(module) != nn.Sequential]
    # return layers[1:]
    layers = []
    temp = [elem for elem in model.children()]

    for layer in temp:
        if isinstance(layer, nn.Sequential):
            for i in layer.children():
                layers.append(i)
        else:
            layers.append(layer)

    return layers


def get_partial_layers(layers, start, end):
    partial_layers = layers[start: end + 1]
    return partial_layers


def get_model_size(model):
    layered_model = get_all_layers(model)
    return len(layered_model)


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
    model = get_alexnet()
    print(get_model_size(model))
    layers = get_all_layers(model)
    partial_layers = get_partial_layers(layers, 0, 20)
    for i in range(len(partial_layers)):
        print(partial_layers[i])






