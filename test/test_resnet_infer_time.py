from dnn_split.model_util import *
from dnn_models.darknet_53 import *
import time



if __name__ == '__main__':
    inputs = get_input()
    print(inputs.size())
    device = torch.device("cpu")
    inputs = inputs.to(device)
    darknet = darknet53(5)
    darknet.to(device)

    times = 100
    total = [0]*13
    for i in range(times):
        res, proc_time, output_size = darknet(inputs)
        total = [a + b for a, b in zip(total, proc_time)]

    for elem in total:
        print(elem)

    for j in output_size:
        print(j)