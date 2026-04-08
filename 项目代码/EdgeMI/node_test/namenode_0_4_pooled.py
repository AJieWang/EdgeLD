import sys
sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up
import torch
import threading
import time
import torch.nn as nn
from VGG.mydefine_VGG16 import VGG_model
from VGG.tensor_op import tensor_divide, tensor_divide_and_fill, tensor_divide_by_computing_and_fill, \
    tensor_divide_by_computing_network_and_fill, tensor_divide_by_computing_and_network
from VGG.tensor_op import merge_total_tensor, merge_part_tensor
from network_and_computing.network_and_computing_record import Network_And_Computing

num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
network_and_computing = Network_And_Computing()
computing_power = network_and_computing.get_computing_power(datanode_num)
network_state = network_and_computing.get_network_state(datanode_num)
computing_a = network_and_computing.get_computing_a(datanode_num)
computing_b = network_and_computing.get_computing_b(datanode_num)

inference_model = VGG_model()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
c_out_list = inference_model.get_c_out()
maxpool_layer = inference_model.get_maxpool_layer()
width = 224
input = torch.rand(1, 3, width, width)
temp_receive_tensor = input
namenode = Network_init_namenode(namenode_num = namenode_num, datanode_num = datanode_num)
thread = []
recv_tensor_list = []
for j in range(datanode_num):
    recv_tensor_list.append(0)
    thread.append(0)

def send_total_data(datanode_name, input_tensor ,start, end, transfer_time_):
    namenode.namenode_send_data(datanode_name = datanode_name, input_tensor = input_tensor, start = start, end = end, transfer_time = transfer_time_)

def send_part_data(datanode_name, input_tensor, start, end):
    namenode.namenode_send_data(datanode_name = datanode_name, input_tensor = input_tensor, start = start, end = end)

def get_end_layer(start = 1, maxpool_layer = []):
    max_value = max(maxpool_layer)
    if start > max_value or start < 1:
        return 0
    for layer in maxpool_layer:
        if layer > start:
            return layer

transfer_time = []
thread_start_time = [0] * datanode_num
thread_end_time = [0] * datanode_num
thread_time = [[] for i in range(datanode_num)]

if __name__ == "__main__":

    time.sleep(3)
    middle_output = input
    final_output = torch.rand(1, 100)
    start_time = time.time()
    
    print("=" * 60)
    print("EdgeLD - 池化层由DataNode计算版本")
    print("DataNode计算每个卷积块的卷积层+池化层")
    print("NameNode只计算全连接层")
    print("=" * 60)
    
    if datanode_num != 1:
        for layer_it in range(1, conv_length + 1, 1):
            if layer_it == conv_length:
                final_output = inference_model(middle_output, layer_it, total_length)
                print("计算全连接层")
            elif layer_it in maxpool_layer:
                print(f"池化层: {layer_it} 由DataNode计算，NameNode跳过")
            elif layer_it == 1 or (layer_it - 1) in maxpool_layer:
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print(f"\n=== 处理卷积块: 层 {start} - {end} (包含池化层 {end}) ===")
                print(f"cross_layer: {cross_layer}")
                
                divided_tensor_list, divide_record = tensor_divide_by_computing_and_network(
                    middle_output, 
                    datanode_num = datanode_num,
                    cross_layer = cross_layer, 
                    computing_power = computing_power,
                    computing_a = computing_a, 
                    computing_b = computing_b,
                    network_state = network_state,
                    c_out = c_out_list[layer_it]
                )
    
                for i in range(datanode_num):
                    thread[i] = threading.Thread(target = send_total_data, args = (i, divided_tensor_list[i], start, end, transfer_time))
                    thread_start_time[i] = time.time()
                    thread[i].start()
                
                for i in range(datanode_num):
                    thread[i].join()
                    thread_end_time[i] = time.time()
                    thread_time[i].append(thread_end_time[i] - thread_start_time[i])
                
                temp = namenode.get_recv_tensor_list()
                middle_output = namenode.get_merged_total_tensor(cross_layer = cross_layer)
                print(f"合并后的 middle_output: {middle_output.size()}")
            else:
                print(f"NameNode不参与第 {layer_it} 层计算")
                continue
    else:
        for layer_it in range(1, conv_length + 1, 1):
            if layer_it == conv_length:
                final_output = inference_model(middle_output, layer_it, total_length)
                print("计算全连接层")
            elif layer_it in maxpool_layer:
                print("池化层由DataNode计算，NameNode跳过")
            elif layer_it == 1 or (layer_it - 1) in maxpool_layer:
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print(f"\n=== 处理卷积块: 层 {start} - {end} (包含池化层 {end}) ===")
    
                for i in range(datanode_num):
                    thread[i] = threading.Thread(target=send_total_data, args=(i, middle_output, start, end, transfer_time))
                    thread[i].start()
                for i in range(datanode_num):
                    thread[i].join()
                temp = namenode.get_recv_tensor_list()
                middle_output = namenode.get_merged_total_tensor(cross_layer = cross_layer)
                print(f"合并后的 middle_output: {middle_output.size()}")
            else:
                print(f"NameNode不参与第 {layer_it} 层计算")
                continue
    
    print("\n" + "=" * 60)
    print("NameNode 结束计算")
    print("Final Output: ", final_output.size())
    end_time = time.time()
    print("Used Time: %0.3fs" % (end_time - start_time))
    print("=" * 60)
    
    # for i in range(datanode_num): 
    #     print('Thread time: %0.3fs, Thread counts: %d' % (sum(thread_time[i]), len(thread_time[i])))
    for i in range(len(transfer_time)): print("transfer_time: ", transfer_time[i])
    print('NameNode Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))
    
    time.sleep(2)
    namenode.close_all()
    print("关闭 NameNode 所有 Socket 连接")
