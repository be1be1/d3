from dnn_split.comm_util import send_model, send_data

if __name__ == '__main__':
    # send_model("../models/partialmodel.pth")
    # send_model("../models/partialmodel2.pth")
    data = list([0,0,0,0,0,0,0])
    print(data)
    send_data(data)