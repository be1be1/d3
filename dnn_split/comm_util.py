import os
import queue
import socket
import struct
import pickle
import threading


def send_model(model_path):
    host = "127.0.0.1"
    port = 50000

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(1)
    s.connect((host, port))

    if os.path.isfile(model_path):
        model_head = struct.pack('128sl', os.path.basename(model_path).encode('utf-8'), os.stat(model_path).st_size)
        s.sendall(model_head)
        f = open(model_path, 'rb')
        print("file opened")
        raw = f.read()
        s.sendall(raw)
        f.close()
    else:
        print("Wrong path.")
    s.close()
    # s.sendall(model_name.encode('utf-8'))
    # s.sendall(os.stat(model_path+model_name).st_size.to_bytes(length=8, byteorder='big'))


def recv_model(model_dir, host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(0)
    s.bind((host, port))
    s.listen()
    print("listening to connection")
    while True:
        conn, addr = s.accept()
        print("connected by ", addr)
        model_info_size = struct.calcsize('128sl')
        buf = conn.recv(model_info_size)
        if buf:
            model_name, model_size = struct.unpack('128sl', buf)
            fn = model_name.decode('utf-8').strip('\00')
            new_model_path = os.path.join(model_dir + fn)
            if model_size == 0:
                continue
            print("model_name:", fn)
            f = open(new_model_path, 'wb')
            while True:
                data = conn.recv(model_size)
                if not data:
                    break
                f.write(data)
            f.close()
        else:
            continue
    s.close()
    # model_name = conn.recv(1024).decode('utf-8')
    # model_size = int.from_bytes(conn.recv(8), byteorder='big')


def send_data(data, host, port):
    # host = "127.0.0.1"
    # port = 50001

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(1)
    s.connect((host, port))
    data_obj = pickle.dumps(data)
    s.sendall(len(data_obj).to_bytes(length=8, byteorder='big'))
    s.sendall(data_obj)
    s.close()
    # s.sendall(model_name.encode('utf-8'))
    # s.sendall(os.stat(model_path+model_name).st_size.to_bytes(length=8, byteorder='big'))


def producer(conn, q):
    size = int.from_bytes(conn.recv(8), byteorder='big')
    data_obj = conn.recv(size)
    data = pickle.loads(data_obj)
    conn.close()
    q.put(item=data, block=False, timeout=10)
    print("I put it into the queue: ", list(q.queue))


def recv_data(q):
    host = "127.0.0.1"
    port = 50001
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(1)
    s.bind((host, port))
    s.listen()
    print("listening to connection")
    while True:
        conn, addr = s.accept()
        print("connected by ", addr)
        producer_thread = threading.Thread(target=producer, args=(conn, q))
        producer_thread.start()


def recv_data_once():
    host = "10.5.27.51"
    port = 50002
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    conn, addr = s.accept()
    print("connected by ", addr)
    size = int.from_bytes(conn.recv(8), byteorder='big')
    data_obj = conn.recv(size)
    data = pickle.loads(data_obj)
    return data


if __name__ == '__main__':
    q = queue.Queue(1000)
    recv_thread = threading.Thread(target=recv_data, args=(q))
    recv_thread.start()
    while True:
        try:
            value = q.get(block=True, timeout=5)
            print("the value is: ", value)
        except queue.Empty:
            print("empty queue")
    # recv_model("../data/models/", "127.0.0.1", 50000)



