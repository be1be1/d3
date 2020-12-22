import torch.nn as nn
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MODEL_PATH = '../data/models/'
IMAGE_PATH = '../data/images/'
#
#
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
#
#
# if __name__ == "__main__":
#     input = get_input()
#     print(input)
#     layer = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True)
#     layer.eval()
#     output = layer(input)
#     print(output)

import torch
import torch.nn as nn

# # With square kernels and equal stride
# m = nn.Conv2d(16, 33, 3, stride=2)
#  # non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# # non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True)
m.eval()
input = get_input()
output = m(input)
print(output)