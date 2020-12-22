from dnn_models.mynet import MyNet
import torch
from torchsummary import summary

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = MyNet().to(device)

    summary(model, (3, 28, 28))