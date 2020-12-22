import threading
from dnn_split.ftp_util import *
from dnn_split.model_canyon import ModelCanyon, ModelFTP
# import pandas as pd
import numpy as np
import torch
# from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class myThread(threading.Thread):
    def __init__(self, threadID, name, input, model):
        super(myThread, self).__init__()
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.input = input
        self.model = model

    def run(self):
        # print("starting" + self.name)
        self.result = perform_partial_forward(self.name, self.input, self.model)
        # print("the size of the output feature map of " + self.name + " is:", self.result.size())
        # print("the output feature map of " + self.name + " is:", output)
        # print("Exiting" + self.name)

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None

def perform_partial_forward(threadName, input, model):
    # alexnet = get_pretrained_alexnet()
    # model = ModelCanyon(model=alexnet, start=0, end=12)
    model.eval()
    output = model(input)

    return output

def output_to_excel(excel_name,output):
    with pd.ExcelWriter(excel_name) as writer:
        for i in range(output.size()[0]):
            for j in range(output.size()[1]):
                data = pd.DataFrame(output[i, j, :, :].detach().numpy())
        #print(data)
                data.to_excel(writer, index=False, header=True, startrow=i*(output.size()[2]+1), startcol=j*output.size()[2])

if __name__ == "__main__":
    threadList = ["Thread_1", "Thread_2", "Thread_3", "Thread_4"]
    threadID = 1
    threads = []
    inputList = []
    result = []
    label = []
    path = "../models/imagenet_classes.txt"

    data_dir = "../data/images/val"

    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with open(path) as f:
        classes = [line.strip() for line in f.readlines()]
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms)
    dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    model = get_pretrained_vgg16()
    # model = get_pretrained_alexnet()
    model.eval()
    class_ori = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataset_loader:
            inputs, labels = data
            # print(labels)
            inputs = inputs.view(1, 3, 224, 224)
            outputs = model(inputs)
            _, index = torch.max(outputs.data, 1)
            percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
            # print("the original predicted class is:")
            # print(classes[index[0]], percentage[index[0]].item())
            class_ori.append(classes[index[0]])
            total += labels.size(0)
            correct += (index == labels).sum().item()

    print('Accuracy of the network on the test images: %d %% without FTP' % (
                    100 * correct / total))

    # test FTP
    # vgg16 = get_pretrained_vgg16()
    # model1 = ModelCanyon(model=vgg16, start=0, end=29)

    # alexnet = get_pretrained_alexnet()
    # # model1 = ModelFTP(model=alexnet, start=0, end=12, x1=0, y1=0, x2=127, y2=127)
    # # model2 = ModelFTP(model=alexnet, start=0, end=12, x1=112, y1=0, x2=224, y2=127)
    # # model3 = ModelFTP(model=alexnet, start=0, end=12, x1=0, y1=112, x2=127, y2=224)
    # # model4 = ModelFTP(model=alexnet, start=0, end=12, x1=112, y1=112, x2=224, y2=224)
    # model1 = ModelFTP(model=alexnet, start=0, end=12, x1=0, y1=0, x2=193, y2=193)
    # model2 = ModelFTP(model=alexnet, start=0, end=12, x1=96, y1=0, x2=224, y2=193)
    # model3 = ModelFTP(model=alexnet, start=0, end=12, x1=0, y1=96, x2=193, y2=224)
    # model4 = ModelFTP(model=alexnet, start=0, end=12, x1=96, y1=96, x2=224, y2=224)
    # model_list = [model1, model2, model3, model4]

    # alexnet = get_pretrained_alexnet()
    vgg16 = get_pretrained_vgg16()
    class_ftp = []
    correct2 = 0
    total2 = 0
    with torch.no_grad():
        for data in dataset_loader:
            inputs, labels = data
            # print(labels)
            threadID = 1
            threads = []
            inputList = []
            result = []

            inputs = inputs.view(1, 3, 224, 224)
            # inputList.append(inputs[:, :, 0:127, 0:127])
            # inputList.append(inputs[:, :, 0:127, 112:224])
            # inputList.append(inputs[:, :, 112:224, 0:127])
            # inputList.append(inputs[:, :, 112:224, 112:224])
            # inputList.append(inputs[:, :, 0:193, 0:193])
            # inputList.append(inputs[:, :, 0:193, 30:224])
            # inputList.append(inputs[:, :, 30:224, 0:193])
            # inputList.append(inputs[:, :, 30:224, 30:224])
            inputList.append(inputs[:, :, 0:130, 0:130])
            inputList.append(inputs[:, :, 0:130, 94:224])
            inputList.append(inputs[:, :, 94:224, 0:130])
            inputList.append(inputs[:, :, 94:224, 94:224])

            # model1 = ModelFTP(model=alexnet, start=0, end=12, x1=0, y1=0, x2=193, y2=193, input_w=inputs.size()[3], input_h=inputs.size()[2])
            # model2 = ModelFTP(model=alexnet, start=0, end=12, x1=30, y1=0, x2=224, y2=193, input_w=inputs.size()[3], input_h=inputs.size()[2])
            # model3 = ModelFTP(model=alexnet, start=0, end=12, x1=0, y1=30, x2=193, y2=224, input_w=inputs.size()[3], input_h=inputs.size()[2])
            # model4 = ModelFTP(model=alexnet, start=0, end=12, x1=30, y1=30, x2=224, y2=224, input_w=inputs.size()[3], input_h=inputs.size()[2])
            model1 = ModelFTP(model=vgg16, start=0, end=16, x1=0, y1=0, x2=130, y2=130, input_w=inputs.size()[3],
                              input_h=inputs.size()[2])
            model2 = ModelFTP(model=vgg16, start=0, end=16, x1=94, y1=0, x2=224, y2=130, input_w=inputs.size()[3],
                              input_h=inputs.size()[2])
            model3 = ModelFTP(model=vgg16, start=0, end=16, x1=0, y1=94, x2=130, y2=224, input_w=inputs.size()[3],
                              input_h=inputs.size()[2])
            model4 = ModelFTP(model=vgg16, start=0, end=16, x1=94, y1=94, x2=224, y2=224, input_w=inputs.size()[3],
                              input_h=inputs.size()[2])
            model_list = [model1, model2, model3, model4]
            for i in range(len(threadList)):
                thread = myThread(threadID, threadList[i], inputList[i], model_list[i])
                thread.start()
                threads.append(thread)
                threadID += 1
                result.append(thread.get_result())

            for t in threads:
                t.join()

            # print("Exiting Main Thread")
            # result.detach().numpy()
            a = np.concatenate((result[0].detach().numpy(), result[1].detach().numpy()), axis=3)
            b = np.concatenate((result[2].detach().numpy(), result[3].detach().numpy()), axis=3)
            c = np.concatenate((a, b), axis=2)
            input2 = torch.from_numpy(c)
            # print(input2.size())
            #
            # modelx = ModelCanyon(model=alexnet, start=13, end=22)
            modelx = ModelCanyon(model=vgg16, start=17, end=50)
            modelx.eval()
            output2 = modelx(input2)
            # # summary(modelx, input_size=(256, 6, 6), batch_size=-1)
            # summary(modelx, input_size=(512, 14, 14), batch_size=-1)
            _, index2 = torch.max(output2, 1)
            percentage2 = torch.nn.functional.softmax(output2, dim=1)[0] * 100
            _, indices2 = torch.sort(output2, descending=True)
            [(classes[idx], percentage[idx].item()) for idx in indices2[0][:5]]
            # print("the FTP predicted class is:")
            # print(classes[index2[0]], percentage2[index2[0]].item())
            class_ftp.append(classes[index2[0]])
            total2 += labels.size(0)
            correct2 += (index2 == labels).sum().item()

        print('Accuracy of the network on the test images: %d %% with FTP' % (
                100 * correct2 / total2))

    match_num = 0
    for i in range(len(class_ftp)):
        # print("the ", i, "pic belongs to class", class_ori[i], "without ftp, and ", class_ftp[i], "with ftp")
        if class_ftp[i] == class_ori[i]:
            match_num += 1

    print("on the 1000 test images, the matching number is:", match_num)

    # class_result = np.vstack((class_ori, class_ftp))
    # np.savetxt('result_imagenet.csv', class_result.T, delimiter=',', header='the inference result', fmt='%s')
