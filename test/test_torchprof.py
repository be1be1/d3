import torch
import torchvision
import torchprof
from dnn_models.mynet import MyNet

model = MyNet()
x = torch.rand([1, 3, 224, 224])

with torchprof.Profile(model, use_cuda=False) as prof:
    model(x)

print(prof.display(show_events=False)) # equivalent to `print(prof)` and `print(prof.display())`