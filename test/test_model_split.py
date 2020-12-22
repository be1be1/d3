from dnn_split.split_point import *

if __name__ == "__main__":

    num_layer = 5
    delay_dev = np.array([4, 3, 5, 8, 6])
    delay_edge = np.array([2, 1, 1, 3, 2])
    data_size = np.array([300, 200, 100, 50])
    bandwidth = 50
    delay_trans = compute_delay_trans(data_size, bandwidth)
    split_point, min_delay = find_split(delay_dev, delay_edge, delay_trans, num_layer)
    print("we split xxx model at: ", split_point+1, "layer to get the minimum inference delay of", min_delay)