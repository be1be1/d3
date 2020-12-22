import torch
import numpy as np
from dnn_split.model_infer_time import *
from dnn_split.split_point import *
from dnn_split.model_util import get_input, get_pretrained_resnet34
from dnn_split.comm_util import recv_data_once

if __name__ == "__main__":
    device = torch.device('cuda')
    input = get_input()
    input = input.to(device)
    print("input finished")
    resnet34 = get_pretrained_resnet34()
    num_layers = get_model_size(resnet34)

    model = ModelInferTimeGPU(model=resnet34, start=0, end=num_layers-1)
    model.to(device)
    model.eval()

    x_size, infer_cloud = model(input)
    print(next(model.parameters()).is_cuda)
    print("Listening to the edge side to receive inference time data ...")
    infer_edge = recv_data_once()
    print("The inference time for each layer on the cloud side is: ", infer_cloud)
    print("The inference time for each layer on the edge side is: ", infer_edge)
    data_size = (np.array(x_size)*4)/pow(10,6)
    print(data_size)
    # output, infer_edge = model(input)
    # infer_cloud = np.array([0.011, 0.0, 0.0002, 0.0001, 0.0, 0.0001, 0.0005, 0.0, 0.0012, 0.0, 0.0005, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.002, 0.0, 0.0, 0.001, 0.0, 0.0005])
    # data_size = np.array([0.7744, 0.7744, 0.186624, 0.559872, 0.559872, 0.129792, 0.259584, 0.259584, 0.173056, 0.173056, 0.173056, 0.173056, 0.036864, 0.036864, 0.009216, 0.004096, 0.004096, 0.004096, 0.004096, 0.004096, 0.001])
    bandwidth = 30  # 6MB/s
    delay_trans = compute_delay_trans(data_size, bandwidth)
    split_point, min_delay = find_split(infer_edge, infer_cloud, delay_trans)
    if split_point == num_layers - 1:
        print("edge side")
    elif split_point == num_layers:
        print("cloud side")
    else:
        print("we split resnet34 model at: ", split_point + 1, "layer to get the minimum inference delay of", min_delay)

    # test_resnet = get_pretrained_resnet34()
    # test_resnet.to(device)
    # test_resnet.eval()
    # torch.cuda.synchronize()
    # start_time = time.time()
    # test_resnet(input)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print("total running time:", end_time-start_time)

