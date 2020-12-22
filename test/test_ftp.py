from dnn_split.fused_tile_patition import *
from dnn_split.ftp_util import *

if __name__ == "__main__":

    # parameters for FTP configuration
    partition_w = 2
    partition_h = 2
    partition = 4
    fused_layer = 8
    task_id = [[0, 1], [2, 3]]
    input_tiles = [[0] * fused_layer for _ in range(partition)]
    output_tiles = [[0] * fused_layer for _ in range(partition)]

    # initialization of each partitioned tile of the bottom layer
    for i in range(partition):
        input_tiles[i][fused_layer-1] = TileRegion(0, 5, 0, 5)
        output_tiles[i][fused_layer-1] = TileRegion(0, 5, 0, 5)

    # get input
    input = get_input()
    input_size = input.size()

    # get DNN model
    model = get_pretrained_alexnet()

    # interpret the model to get relevant parameters for FTP algo
    model = ModelInterpreter(model=model)
    x_size, x_kenerl_size, x_stride, x_padding, x_type = model(input)
    output_width = x_size[-1][2]
    output_height = x_size[-1][1]

    # perform FTP algo
    net_para = load_dnn_model(input_size[1:], x_size, x_kenerl_size, x_stride, x_padding, x_type)
    ftp_para = FtpPara(partition_w, partition_h, fused_layer, task_id, input_tiles, output_tiles)
    ftp_para = perform_ftp(net_para, ftp_para, output_width, output_height)

    # print the coordinate of each partitioned tile for each layer
    print("we partition each layer of the DNN model into ", partition, "parts:")
    for i in range(partition_h):
        for j in range(partition_w):
            for l in range(fused_layer):
                id = ftp_para.task_id[i][j]
                print("input Layer", l + 1, " :", "coordination of the ", id + 1, "part: (",
                      ftp_para.input_tiles[id][l].top_left_x, ",",
                      ftp_para.input_tiles[id][l].top_left_y, "),(", ftp_para.input_tiles[id][l].bottom_right_x, ",",
                      ftp_para.input_tiles[id][l].bottom_right_y, ")")
                print("output Layer", l + 1, " :", "coordination of the ", id + 1, "part: (", ftp_para.output_tiles[id][l].top_left_x,
                      ",",
                      ftp_para.output_tiles[id][l].top_left_y, "),(", ftp_para.output_tiles[id][l].bottom_right_x, ",",
                      ftp_para.output_tiles[id][l].bottom_right_y, ")")
            print("----------------------------------------------------------------")
