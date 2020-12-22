from dnn_split.model_infer_time import *
from dnn_split.model_util import get_input, get_pretrained_resnet34
from dnn_split.comm_util import send_data


if __name__ == "__main__":
    input = get_input()

    resnet34 = get_pretrained_resnet34()
    num_layers = get_model_size(resnet34)

    model = ModelInferTime(model=resnet34, start=0, end=num_layers-1)
    model.eval()

    data_size, infer_edge = model(input)
    send_data(infer_edge, "10.5.27.51", 50002)
    print(sum(infer_edge))
    # send_data(infer_edge, "127.0.0.1", 50002)



    # infer_edge = recv_data_once()
    # data_size = np.array(data_size)
    # infer_cloud = np.array(infer_cloud)
    # infer_edge = np.array(infer_edge)
    # # output, infer_edge = model(input)
    # # infer_cloud = np.array([0.011, 0.0, 0.0002, 0.0001, 0.0, 0.0001, 0.0005, 0.0, 0.0012, 0.0, 0.0005, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.002, 0.0, 0.0, 0.001, 0.0, 0.0005])
    # # data_size = np.array([0.7744, 0.7744, 0.186624, 0.559872, 0.559872, 0.129792, 0.259584, 0.259584, 0.173056, 0.173056, 0.173056, 0.173056, 0.036864, 0.036864, 0.009216, 0.004096, 0.004096, 0.004096, 0.004096, 0.004096, 0.001])
    # bandwidth = 6  # 6MB/s
    # delay_trans = compute_delay_trans(data_size, bandwidth)
    # split_point, min_delay = find_split(infer_edge, infer_cloud, delay_trans)
    # if split_point == num_layers - 1:
    #     print("edge side")
    # elif split_point == num_layers:
    #     print("cloud side")
    # else:
    #     print("we split xxx model at: ", split_point + 1, "layer to get the minimum inference delay of", min_delay)

