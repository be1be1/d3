import threading
from dnn_split.ftp_util import *
from dnn_split.model_canyon import ModelCanyon
from dnn_split.model_ftp import ModelFTP

MODEL_PATH = '../data/models/'
IMAGE_PATH = '../data/images/'

class myThread(threading.Thread):
    def __init__(self, threadID, name, input, model):
        super(myThread, self).__init__()
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.input = input
        self.model = model

    def perform_partial_forward(self):
        self.model.eval()
        output = self.model(self.input)

        return output

    def run(self):
        self.result = self.perform_partial_forward()

    def get_result(self):
        threading.Thread.join(self)
        try:
            return self.result
        except Exception:
            return None

if __name__ == "__main__":

    input = get_input()
    # multi-thread
    threadList = ["Thread_1", "Thread_2", "Thread_3", "Thread_4"]
    threadID = 1
    threads = []
    inputList = []
    result = []

    # original inference without fused tile partition
    alexnet0 = get_pretrained_alexnet()
    model = ModelCanyon(alexnet0, 0, 12)
    model.eval()
    output = model(input)

    print("the output:")
    print(output)
    print("-------------------------------------------")

    # inference with fused tile partition
    alexnet = get_pretrained_alexnet()

    # partition the input feature map into four parts
    # the coordinate of each partitioned tile of the top layer
    coordinate_1 = TileRegion(0, 0, 193, 193)
    coordinate_2 = TileRegion(30, 0, 224, 193)
    coordinate_3 = TileRegion(0, 30, 193, 224)
    coordinate_4 = TileRegion(30, 30, 224, 224)
    # each partitioned tile completes inference separately
    model_1 = ModelFTP(model=alexnet, start=0, end=12, coordinate=coordinate_1, input_w=input.size()[3],
                      input_h=input.size()[2])
    model_2 = ModelFTP(model=alexnet, start=0, end=12, coordinate=coordinate_2, input_w=input.size()[3],
                      input_h=input.size()[2])
    model_3 = ModelFTP(model=alexnet, start=0, end=12, coordinate=coordinate_3, input_w=input.size()[3],
                      input_h=input.size()[2])
    model_4 = ModelFTP(model=alexnet, start=0, end=12, coordinate=coordinate_4, input_w=input.size()[3],
                      input_h=input.size()[2])
    model_list = [model_1, model_2, model_3, model_4]

    inputList.append(input[:, :, 0:193, 0:193])
    inputList.append(input[:, :, 0:193, 30:224])
    inputList.append(input[:, :, 30:224, 0:193])
    inputList.append(input[:, :, 30:224, 30:224])

    for i in range(len(threadList)):
        thread = myThread(threadID, threadList[i], inputList[i], model_list[i])      # use multi thread to compute in parallel
        thread.start()
        threads.append(thread)
        threadID += 1
        result.append(thread.get_result())

    for t in threads:
        t.join()

    a = np.concatenate((result[0].detach().numpy(), result[1].detach().numpy()), axis=3)
    b = np.concatenate((result[2].detach().numpy(), result[3].detach().numpy()), axis=3)
    c = np.concatenate((a, b), axis=2)
    output_1 = torch.from_numpy(c)
    print(output_1)

