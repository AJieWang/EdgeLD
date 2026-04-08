import sys
sys.path.append("../..")
sys.path.append("..")

import torch, time, socket, json, six
import torch.nn as nn
import numpy as np
from VGG.tensor_op import merge_total_tensor, merge_part_tensor


# IP设置
# namenode_ip = "127.0.0.1"
namenode_ip = "192.168.202.129"
# datanode_ip = ["127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1", "127.0.0.1"]
datanode_ip = ["192.168.202.130", "192.168.202.131", "192.168.202.132", "192.168.202.133", "192.168.202.134", "192.168.202.135"]
datanode_port = [10000, 10001, 10002, 10003, 10004, 10005]

namenode_pre_send = []
datanode_pre_send = []
after_receive = []

# 数据量阈值 (10MB)
DATA_SIZE_THRESHOLD = 10 * 1024 * 1024


def get_tensor_size(tensor):
    """计算tensor大小(bytes), float32为4字节"""
    return tensor.numel() * 4


def adaptive_send_strategy(socket, input_tensor, datanode_name, memory_state, transfer_time):
    """自适应传输策略
    
    根据数据量和节点内存状态决定传输策略:
    - 数据量 < 10MB: 直接传输
    - 数据量 >= 10MB 且 内存不足: 分块传输
    - 数据量 >= 10MB 但内存足够: 压缩后传输 (当前简化版本仍用直接传输)
    """
    tensor_size = get_tensor_size(input_tensor)
    available_memory = memory_state[datanode_name] * 0.5 if datanode_name < len(memory_state) else float('inf')
    
    print(f"DataNode {datanode_name}: 数据大小 {tensor_size/1e6:.1f}MB, 可用内存 {available_memory/1e6:.1f}MB")
    
    if tensor_size < DATA_SIZE_THRESHOLD:
        # 策略1: 直接传输
        print(f"DataNode {datanode_name}: 使用直接传输策略")
        return direct_send(socket, input_tensor, transfer_time)
    elif tensor_size * 1.5 > available_memory:
        # 策略2: 数据过大且内存不足，分块传输
        print(f"DataNode {datanode_name}: 使用分块传输策略")
        return chunked_send(socket, input_tensor, transfer_time)
    else:
        # 策略3: 数据量大但内存足够 (简化版本，仍用直接传输)
        print(f"DataNode {datanode_name}: 数据较大但内存足够，使用直接传输策略")
        return direct_send(socket, input_tensor, transfer_time)


def direct_send(socket, input_tensor, transfer_time):
    """直接发送数据"""
    pre_send_time = time.time()
    input_numpy = input_tensor.detach().numpy()
    
    start = str(0).encode(encoding="UTF-8")
    end = str(0).encode(encoding="UTF-8")
    input_numpy_size = get_numpy_size(input_tensor)
    input_bytes = input_numpy.tobytes()
    
    send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
    send_data_len = str(len(send_data)).encode(encoding="UTF-8")
    socket.send(send_data_len)
    time.sleep(0.01)
    
    transfer_start_time = time.time()
    temp_time = transfer_start_time - pre_send_time
    namenode_pre_send.append(temp_time)
    
    socket.sendall(send_data)
    transfer_time.append(time.time() - transfer_start_time)


def chunked_send(socket, input_tensor, transfer_time):
    """分块传输数据 (将数据分成多个小块传输)"""
    pre_send_time = time.time()
    
    # 分块大小 5MB
    chunk_size = 5 * 1024 * 1024
    input_numpy = input_tensor.detach().numpy()
    total_size = input_numpy.nbytes
    num_chunks = (total_size + chunk_size - 1) // chunk_size
    
    print(f"分块传输: 共 {num_chunks} 块, 每块 {chunk_size/1e6:.1f}MB")
    
    # 先发送分块数量
    chunk_info = str(num_chunks).encode(encoding="UTF-8")
    socket.send(chunk_info)
    time.sleep(0.01)
    
    # 分块发送
    transfer_start_time = time.time()
    for i in range(num_chunks):
        start_byte = i * chunk_size
        end_byte = min((i + 1) * chunk_size, total_size)
        chunk_data = input_numpy.tobytes()[start_byte:end_byte]
        
        chunk_len = str(len(chunk_data)).encode(encoding="UTF-8")
        socket.send(chunk_len)
        socket.sendall(chunk_data)
        
        print(f"已发送块 {i+1}/{num_chunks}")
    
    transfer_time.append(time.time() - transfer_start_time)


class Network_init_namenode():
    def __init__(self, namenode_num = 1, datanode_num = 1):
        super(Network_init_namenode, self).__init__()
        print ("NameNode 开始初始化")
        self.datanode_num = datanode_num
        self.client_socket = []
        if (datanode_num >= 1):
            for hostname in range(datanode_num):
                tcp_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                tcp_client_socket.connect((datanode_ip[hostname], datanode_port[hostname]))
                hello_world = "Hello DataNode "+ str(hostname) + ", I'm NameNode"
                tcp_client_socket.send(hello_world.encode())

                recv_data_test = tcp_client_socket.recv(1024)
                print (str(recv_data_test, encoding="UTF-8"))
                self.client_socket.append(tcp_client_socket)
        print ("NameNode 初始化完成")
        self.recv_tensor_temp_list = []
        for it in range(datanode_num):
            self.recv_tensor_temp_list.append(0)

    def get_recv_tensor_list(self):
        return self.recv_tensor_temp_list

    def get_merged_total_tensor(self, divide_record = 0, cross_layer = 1):
        temp = merge_total_tensor(self.recv_tensor_temp_list, divide_record = divide_record, cross_layer = cross_layer)
        return temp

    def get_merged_part_tensor(self):
        temp = merge_part_tensor(self.recv_tensor_temp_list, divide_record=0, cross_layer=1)
        return temp

    def namenode_send_data(self, datanode_name, input_tensor, start, end, transfer_time, memory_state=None):
        """发送数据，包含内存状态参数用于自适应传输"""
        if memory_state is None:
            memory_state = [1e9] * self.datanode_num
        
        pre_send_time = time.time()
        input_numpy = input_tensor.detach().numpy()
        start = str(start).encode(encoding='utf-8')
        end = str(end).encode(encoding='utf-8')
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()
        send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
        send_data_len = str(len(send_data)).encode(encoding='utf-8')
        self.client_socket[datanode_name].send(send_data_len)
        time.sleep(0.01)

        transfer_start_time = time.time()
        temp_time = transfer_start_time - pre_send_time
        namenode_pre_send.append(temp_time)
        print('NameNode Pre send time: %0.3fs, Total pre send time: %0.3fs' % (temp_time, sum(namenode_pre_send)))

        self.client_socket[datanode_name].sendall(send_data)

        transfer_time.append(time.time() - transfer_start_time)

        data_total_len = self.client_socket[datanode_name].recv(1024)
        data_total_len = int(str(data_total_len, encoding="UTF-8"))
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.client_socket[datanode_name].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        after_receive_start_time = time.time()
        split_list = recv_data.split(b'@#$%')
        recv_start = int(str(split_list[0], encoding="UTF-8"))
        recv_end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        recv_numpy = np.frombuffer(split_list[3], dtype = np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        self.recv_tensor_temp_list[datanode_name] = recv_tensor

        temp_time = time.time() - after_receive_start_time
        after_receive.append(temp_time)
        print('NameNode After receive time: %0.3fs, Total after receive time: %0.3fs' % (temp_time, sum(after_receive)))
        return recv_start, recv_end, recv_tensor

    def namenode_recv_data(self, datanode_name):
        data_total_len = self.client_socket[datanode_name].recv(1024)
        data_total_len = int(str(data_total_len, encoding='utf-8'))
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.client_socket[datanode_name].recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        split_list = recv_data.split(b'@#$%')
        start = int(str(split_list[0], encoding="UTF-8"))
        end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        recv_numpy = np.frombuffer(split_list[3], dtype=np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape=recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)
        return start, end, recv_tensor

    def close(self, datanode_name):
        self.client_socket[datanode_name].close()
    def close_all(self):
        for i in range(self.datanode_num):
            self.client_socket[i].close()

class Network_init_datanode():
    def __init__(self, namenode_num = 1, datanode_num = 3, datanode_name = 0):
        super(Network_init_datanode, self).__init__()
        print ("DataNode %d 开始初始化" % datanode_name )
        tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_server_socket.bind((datanode_ip[datanode_name], datanode_port[datanode_name]))
        tcp_server_socket.listen(2)
        self.datanode_socket, client_addr = tcp_server_socket.accept()
        recv_data_test = self.datanode_socket.recv(1024)
        print (str(recv_data_test, encoding="UTF-8"))
        hello_world = "Hello NameNode, I have received your hello world, I'm DateNode " + str(datanode_name)
        self.datanode_socket.send(hello_world.encode())
        print("DataNode %d 初始化完成\n" % datanode_name)

        self.datanode_num = datanode_num
        self.datanode_name = datanode_name
        self.last_inference_layer = 0
        self.saved_tensor = torch.rand(1, 1, 1, 1)
        self.divied_tensor_list = []
        if datanode_name==0 or datanode_name == datanode_num-1:
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
        else:
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
            self.divied_tensor_list.append(torch.rand(1, 1, 1, 1))
    def set_last_inference_layer(self, layer):
        self.last_inference_layer = layer
    def set_saved_tensor(self, tensor):
        self.saved_tensor = tensor
    def set_divied_tensor_list(self, tensor_list):
        self.divied_tensor_list = tensor_list
    def get_last_inference_layer(self):
        return int(self.last_inference_layer)
    def get_saved_tensor(self):
        return self.saved_tensor
    def get_divied_merged_tensor(self):
        if self.datanode_name == 0 or self.datanode_num - 1:
            return self.divied_tensor_list[0]
        else:
            return torch.cat((self.divied_tensor_list[0], self.divied_tensor_list[1]), 3)
    def get_merged_tensor(self):
        if self.datanode_name == 0:
            merged_tensor = torch.cat((self.saved_tensor, self.divied_tensor_list[0]), 3)
        elif self.datanode_name == self.datanode_num - 1:
            merged_tensor = torch.cat((self.divied_tensor_list[0], self.saved_tensor), 3)
        else:
            merged_tensor = torch.cat((self.divied_tensor_list[0], self.saved_tensor, self.divied_tensor_list[1]), 3)
        return merged_tensor
    def empty_tensor(self):
        self.saved_tensor = torch.rand(1, 1, 1, 1)
        if self.datanode_name==0 or self.datanode_name == self.datanode_num-1:
            self.divied_tensor_list[0] = torch.rand(1, 1, 1, 1)
        else:
            self.divied_tensor_list[0] = torch.rand(1, 1, 1, 1)
            self.divied_tensor_list[1] = torch.rand(1, 1, 1, 1)


    def datanode_send_data(self, input_tensor, transfer_time, start=0, end=0):
        pre_send_time = time.time()
        input_numpy = input_tensor.detach().numpy()

        start = str(start).encode(encoding="UTF-8")
        end = str(end).encode(encoding="UTF-8")
        input_numpy_size = get_numpy_size(input_tensor)
        input_bytes = input_numpy.tobytes()

        send_data = start + b'@#$%' + end + b'@#$%' + input_numpy_size + b'@#$%' + input_bytes
        send_data_len = str(len(send_data)).encode(encoding="UTF-8")
        self.datanode_socket.send(send_data_len)
        time.sleep(0.01)
        temp_time = time.time() - pre_send_time
        datanode_pre_send.append(temp_time)
        print('DataNode Pre send time: %0.3fs, Total pre send time: %0.3fs' % (temp_time, sum(datanode_pre_send)))

        transfer_start_time = time.time()

        self.datanode_socket.sendall(send_data)

        transfer_time.append(time.time() - transfer_start_time)

    def datanode_recv_data(self, pre_conv):
        data_total_len = b''
        while True:
            data = self.datanode_socket.recv(1024)
            if len(data) != 0:
                data_total_len = data
                break
        pre_conv_start_time = time.time()

        print ("DataNode recv data length: ", str(data_total_len, encoding='utf-8'))
        data_total_len = int(str(data_total_len, encoding="UTF-8"))
        print("DataNode recv data length: ", data_total_len)
        recv_data_len = 0
        recv_data = b''
        while recv_data_len < data_total_len:
            recv_data_temp = self.datanode_socket.recv(10240)
            recv_data_len += len(recv_data_temp)
            recv_data += recv_data_temp

        split_list = recv_data.split(b'@#$%')
        start = int(str(split_list[0], encoding="UTF-8"))
        end = int(str(split_list[1], encoding="UTF-8"))

        recv_numpy_size = get_recv_tensor_size(split_list[2])
        recv_numpy = np.frombuffer(split_list[3], dtype = np.float32)
        recv_numpy = np.reshape(recv_numpy, newshape = recv_numpy_size)
        recv_tensor = torch.from_numpy(recv_numpy)

        pre_conv.append(time.time() - pre_conv_start_time)
        return start, end, recv_tensor

    def close(self):
        self.datanode_socket.close()

def get_recv_tensor_size(split_list_bytes):
    split_str = str(split_list_bytes, encoding="UTF-8").split("*")
    recv_numpy_size = []
    for i ,value in enumerate(split_str):
        recv_numpy_size.append(int(value))
    return tuple(recv_numpy_size)

def get_numpy_size(input_tensor):
    size_list = list(input_tensor.size())
    input_numpy_size = ""
    length = len(size_list)
    for i, value in enumerate(size_list):
        if i == length - 1:
            input_numpy_size += str(value)
        else:
            input_numpy_size += str(value)
            input_numpy_size += "*"
    return input_numpy_size.encode(encoding="UTF-8")


