from dnn_split.ftp_util import *
import numpy as np

def grid(output_width, output_height, ftp_para, partition_w, partition_h):
    """calculate the coordination of each partition for the bottom layer
    Args:
        output_width: width of the output feature map of the bottom layer
        output_height: height of the output feature map of the bottom layer
        ftp_para: initialized ftp para  of the bottom layer updated
        partition_w: the number of slices divided from width
        partition_h: the number of slices divided from height

    Returns:
        ftp_para: ftp para of the bottom layer updated for FTP algorithm
    """
    w = output_width
    h = output_height
    stride_w = np.ceil(w/partition_w)
    stride_h = np.ceil(h/partition_h)
    start_h = 0
    end_h = stride_h

    for i in range(partition_h):
        start_w = 0
        end_w = stride_w
        if i != 0:
            start_h = start_h + stride_h
            end_h = end_h + stride_h
        for j in range(partition_w):
            task_id = ftp_para.task_id[i][j]
            ftp_para.output_tiles[task_id][ftp_para.fused_layers-1].top_left_x = start_w
            ftp_para.output_tiles[task_id][ftp_para.fused_layers-1].bottom_right_x = end_w
            ftp_para.output_tiles[task_id][ftp_para.fused_layers-1].top_left_y = start_h
            ftp_para.output_tiles[task_id][ftp_para.fused_layers-1].bottom_right_y = end_h
            start_w = end_w
            if j == partition_w - 1:
                end_w = w
            else:
                end_w = end_w + stride_w

    return ftp_para

def tranversal(net_para, output):
    """calculate the coordination of the partitioned tile for current layer
    Args:
        net_para: net para of the current layer
        output: TileRegion info of the output partitioned tile for current layer

    Returns:
        input: TileRegion info of the input partitioned tile for current layer
    """
    input = TileRegion(0, 0, 0, 0)
    stride = net_para.stride
    kernel_size = net_para.kernel_size
    padding = net_para.padding
    input_w = net_para.input_width
    input_h = net_para.input_height

    # calculate the coordination of the input partitioned tiles for current layer
    if net_para.type == "convolution" or net_para.type == "pooling":
        input.top_left_x = output.top_left_x * stride
        input.top_left_y = output.top_left_y * stride
        input.bottom_right_x = (output.bottom_right_x - 1) * stride + kernel_size
        input.bottom_right_y = (output.bottom_right_y - 1) * stride + kernel_size

    # update the coordination of the input partitioned tile considering different situations with padding effect
    if input.bottom_right_x == input_w + 2 * padding and input.bottom_right_y == input_h + 2 * padding:   # the partitioned tile locates at the bottom right corner of the feature map
        input.top_left_x = max(0, input.top_left_x - padding)
        input.top_left_y = max(0, input.top_left_y - padding)
        input.bottom_right_x = input.bottom_right_x - 2 * padding
        input.bottom_right_y = input.bottom_right_y - 2 * padding
    elif input.bottom_right_x == input_w + 2 * padding:        # the partitioned tile locates at the right side of the feature map
        input.top_left_x = max(0, input.top_left_x - padding)
        input.top_left_y = max(0, input.top_left_y - padding)
        input.bottom_right_x = input.bottom_right_x - 2 * padding
        input.bottom_right_y = input.bottom_right_y - padding
    elif input.bottom_right_y == input_h + 2 * padding:    # the partitioned tile locates at the down side of the feature map
        input.top_left_x = max(0, input.top_left_x - padding)
        input.top_left_y = max(0, input.top_left_y - padding)
        input.bottom_right_x = input.bottom_right_x - padding
        input.bottom_right_y = input.bottom_right_y - 2 * padding
    else:
        input.top_left_x = max(0, input.top_left_x - padding)
        input.top_left_y = max(0, input.top_left_y - padding)
        input.bottom_right_x = max(0, input.bottom_right_x - padding)
        input.bottom_right_y = max(0, input.bottom_right_y - padding)

    return input


def perform_ftp(net_para, ftp_para, output_width, output_height):
    """perform FTP algorithm
    Args:
        net_para: net para of the DNN model
        ftp_para: initialized para for FTP algorithm
        output_width: width of the output feature map of the bottom layer
        output_height: height of the output feature map of the bottom layer

    Returns:
        ftp_para: updated ftp para, which gives the coordination of each partitioned tile for each layer
    """
    id = 0
    for i in range(ftp_para.partitions_h):
        for j in range(ftp_para.partitions_w):
            ftp_para.task_id[i][j] = id
            id += 1

    grid(output_width, output_height, ftp_para, ftp_para.partitions_w, ftp_para.partitions_h)
    for i in range(ftp_para.partitions_h):
        for j in range(ftp_para.partitions_w):
            for l in range(ftp_para.fused_layers-1, -1, -1):
                ftp_para.input_tiles[ftp_para.task_id[i][j]][l] = tranversal(net_para[l], ftp_para.output_tiles[ftp_para.task_id[i][j]][l])    #derive the coordination from the bottom layer
                if l > 0:
                    ftp_para.output_tiles[ftp_para.task_id[i][j]][l-1] = ftp_para.input_tiles[ftp_para.task_id[i][j]][l]   # assign the input tiles of current layer as the output tiles of the previous layer

    return ftp_para
