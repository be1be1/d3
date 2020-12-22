import numpy as np
import joblib


def find_split(delay_edge, delay_cloud, delay_trans):
    num = len(delay_cloud)
    total_delay = np.zeros(num+1)
    for i in range(num-1):
        total_delay[i] = np.sum(delay_edge[0:i+1]) + np.sum(delay_cloud[i+1:num]) + delay_trans[i]
    total_delay[num-1] = np.sum(delay_edge)
    total_delay[num] = np.sum(delay_cloud) + delay_trans[num-1]
    split_point = np.argmin(total_delay)
    min_delay = np.min(total_delay)
    return split_point, min_delay


def compute_delay_trans(data_size, bandwidth):
    delay_trans = data_size/bandwidth
    return delay_trans


def predict_delay_per_layer(layer_type, layer_conf_para, dev_info):
    PATH = "../model/regression_model.m"
    input = [layer_type, layer_conf_para, dev_info]
    model = joblib.load(PATH)
    delay_per_layer = model.predict(input)
    return delay_per_layer
