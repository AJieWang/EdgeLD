import sys
sys.path.append("../..")
sys.path.append("..")

from node_test.network_op_v2 import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up
import torch
import threading
import time
import torch.nn as nn
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op_v2 import tensor_divide, tensor_divide_and_fill, tensor_divide_by_computing_and_fill, \
    tensor_divide_by_computing_network_and_fill, tensor_divide_by_computing_and_network, \
    tensor_divide_by_computing_network_and_memory
from VGG.tensor_op_v2 import merge_total_tensor, merge_part_tensor
from network_and_computing.network_and_computing_record_v2 import Network_And_Computing

num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
network_and_computing = Network_And_Computing()
computing_power = network_and_computing.get_computing_power(datanode_num)
network_state = network_and_computing.get_network_state(datanode_num)
computing_a = network_and_computing.get_computing_a(datanode_num)
computing_b = network_and_computing.get_computing_b(datanode_num)
memory_state = network_and_computing.get_memory_state(datanode_num)

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

def send_total_data(datanode_name, input_tensor ,start, end, transfer_time_, memory_state_):
    namenode.namenode_send_data(datanode_name = datanode_name, input_tensor = input_tensor, 
                                 start = start, end = end, transfer_time = transfer_time_,
                                 memory_state = memory_state_)

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

    print("="*60)
    print("EdgeLD V2 - 内存约束自适应传输优化")
    print("="*60)
    print(f"DataNode 数量: {datanode_num}")
    print(f"内存状态: {[f'{m/1e9:.1f}GB' for m in memory_state]}")
    print(f"计算能力: {computing_power}")
    print(f"网络带宽: {[f'{n/1e6:.0f}Mbps' for n in network_state]}")
    print("="*60)

    time.sleep(3)
    middle_output = input
    start_time = time.time()
    
    for layer_it in range(1, conv_length + 1, 1):
        if layer_it == conv_length:
            final_output = inference_model(middle_output, layer_it, layer_it)
        elif layer_it in maxpool_layer:
            middle_output = inference_model(middle_output, layer_it, layer_it)
        else:
            # 使用新增的内存约束划分函数 (V2版本)
            divided_tensor, divide_record = tensor_divide_by_computing_network_and_memory(
                middle_output,
                datanode_num = datanode_num,
                cross_layer = 1,
                computing_power = computing_power,
                computing_a = computing_a,
                computing_b = computing_b,
                network_state = network_state,
                memory_state = memory_state,
                c_out = c_out_list[layer_it]
            )
            
            for i in range(datanode_num):
                thread[i] = threading.Thread(target = send_total_data, 
                                             args = (i, divided_tensor[i], layer_it, layer_it, transfer_time, memory_state))
                thread_start_time[i] = time.time()
                thread[i].start()
            
            for i in range(datanode_num):
                thread[i].join()
                thread_end_time[i] = time.time()
                thread_time[i].append(thread_end_time[i] - thread_start_time[i])

            merged_tensor = namenode.get_merged_total_tensor()
            middle_output = merged_tensor
            
        print ("middle_output: ", middle_output.size())
        print ("结束 %d的推理" % layer_it)
        
    end_time = time.time()
    print("Used time: %0.3fs" % (end_time - start_time))

    for i in range(datanode_num): 
        print('Thread time: %0.3fs, Thread counts: %d' % (sum(thread_time[i]), len(thread_time[i])))

    print('NameNode Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))

    time.sleep(2)
    namenode.close_all()
    print("关闭 NameNode 所有 Socket 连接")
