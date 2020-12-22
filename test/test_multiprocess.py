import threading
from dnn_split.ftp_util import *
from dnn_split.model_canyon import ModelCanyon
import pandas as pd
import numpy as np
import torch
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# import torch.multiprocessing as mp
# from torch.multiprocessing import Pool, Manager
import multiprocessing
from multiprocessing import Pool
import time

# class myProcess(multiprocessing.Process):
#     def __init__(self, func, args):
#         multiprocessing.Process.__init__(self)
#         self.func = func
#         self.args = args
#         super(myProcess, self).__init__()
#
#     def run(self):
#         # print("starting" + self.name)
#         self.result = self.func(*self.args)
#         # print("the size of the output feature map of " + self.name + " is:", self.result.size())
#         # print("the output feature map of " + self.name + " is:", output)
#         # print("Exiting" + self.name)
#
#     def get_result(self):
#         multiprocessing.Process.join(self)
#         try:
#             return self.result
#         except Exception:
#             return None

def perform_partial_forward(input, model):
    # alexnet = get_pretrained_alexnet()
    # model = ModelCanyon(model=alexnet, start=0, end=12)
    model.eval()
    output = model(input)

    return output

if __name__ == "__main__":
    processList = ["Process_1", "Process_2", "Process_3", "Process_4"]
    label = []
    path = "../models/imagenet_classes.txt"

    data_dir = "../data/images/val"

    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # dataset = datasets.ImageFolder(data_dir, transform=transforms)
    with open(path) as f:
        classes = [line.strip() for line in f.readlines()]
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # model = get_pretrained_vgg16()
    # # model = get_pretrained_alexnet()
    # model.eval()
    # class_ori = []
    # correct = 0
    # total = 0
    # start_time = time.time()
    # with torch.no_grad():
    #     for data in dataset_loader:
    #         inputs, labels = data
    #         # print(labels)
    #         inputs = inputs.view(1, 3, 224, 224)
    #         outputs = model(inputs)
    #         # _, predicted = torch.max(outputs.data, 1)
    #         _, index = torch.max(outputs.data, 1)
    #         percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    #         # print("the original predicted class is:")
    #         # print(classes[index[0]], percentage[index[0]].item())
    #         class_ori.append(classes[index[0]])
    #         total += labels.size(0)
    #         correct += (index == labels).sum().item()
    # end_time = time.time()
    # print('Inference process cost:', end_time - start_time, "s")
    # print('Accuracy of the network on the test images: %d %% without FTP' % (
    #                 100 * correct / total))

    # test FTP
    vgg16 = get_pretrained_vgg16()
    model1 = ModelCanyon(model=vgg16, start=0, end=29)

    # alexnet = get_pretrained_alexnet()
    # model1 = ModelCanyon(model=alexnet, start=0, end=12)

    class_ftp = []
    correct2 = 0
    total2 = 0
    multiprocessing.freeze_support()
    start_time = time.time()
    with torch.no_grad():
        for data in dataset_loader:
            inputs, labels = data
            # print(labels)
            # processID = 1
            # processNum = []
            inputList = []
            pool = multiprocessing.Pool()
            result = []

            inputs = inputs.view(1, 3, 224, 224)
            inputList.append(inputs[:, :, 0:127, 0:127])
            inputList.append(inputs[:, :, 0:127, 112:224])
            inputList.append(inputs[:, :, 112:224, 0:127])
            inputList.append(inputs[:, :, 112:224, 112:224])
            # inputList.append(inputs[:, :, 0:127, 0:127])
            # inputList.append(inputs[:, :, 0:127, 93:224])
            # inputList.append(inputs[:, :, 93:224, 0:127])
            # inputList.append(inputs[:, :, 93:224, 93:224])

            for i in range(len(processList)):
                # process = myProcess(func=perform_partial_forward, args=(inputList[i], model1, return_dict))
                return_result = pool.apply_async(perform_partial_forward, args=(inputList[i], model1))
                result.append(return_result)

            pool.close()
            pool.join()

            a = np.concatenate((result[0].get().detach().numpy(), result[1].get().detach().numpy()), axis=3)
            b = np.concatenate((result[2].get().detach().numpy(), result[3].get().detach().numpy()), axis=3)
            c = np.concatenate((a, b), axis=2)
            input2 = torch.from_numpy(c)
            #
            # model2 = ModelCanyon(model=alexnet, start=13, end=22)
            model2 = ModelCanyon(model=vgg16, start=30, end=50)
            model2.eval()
            output2 = model2(input2)
            # # summary(model2, input_size=(256, 6, 6), batch_size=-1)
            # summary(model2, input_size=(512, 14, 14), batch_size=-1)
            _, index2 = torch.max(output2, 1)
            percentage2 = torch.nn.functional.softmax(output2, dim=1)[0] * 100
            # _, indices2 = torch.sort(output2, descending=True)
            # [(classes[idx], percentage[idx].item()) for idx in indices2[0][:5]]
            # print("the FTP predicted class is:")
            # print(classes[index2[0]], percentage2[index2[0]].item())
            class_ftp.append(classes[index2[0]])
            total2 += labels.size(0)
            correct2 += (index2 == labels).sum().item()
    end_time = time.time()
    print('Inference with FTP cost:', end_time-start_time, "s")
    print('Accuracy of the network on the test images: %d %% with FTP' % (
                100 * correct2 / total2))

    # match_num = 0
    # for i in range(len(class_ftp)):
    #     # print("the ", i, "pic belongs to class", class_ori[i], "without ftp, and ", class_ftp[i], "with ftp")
    #     if class_ftp[i] == class_ori[i]:
    #         match_num += 1
    #
    # print("on the 1000 test images, the matching number is:", match_num)
    #
    # class_result = np.vstack((class_ori, class_ftp))
    # np.savetxt('result_imagenet.csv', class_result.T, delimiter=',', header='the inference result', fmt='%s')
