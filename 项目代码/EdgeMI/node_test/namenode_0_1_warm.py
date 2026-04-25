import sys
sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up, VGG_model, sample_tensor
import torch
import threading
import time
import torch.nn as nn
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

inference_model = VGG_model()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
maxpool_layer = inference_model.get_maxpool_layer()

WARM_UP_ROUNDS = 3
VALID_ROUNDS = 2
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS


def send_total_data(datanode_name, input_tensor, start, end, transfer_time_):
    # 发送 total 数据
    try:
        namenode.namenode_send_data(
            datanode_name=datanode_name,
            input_tensor=input_tensor,
            start=start,
            end=end,
            transfer_time=transfer_time_
        )
    except Exception as e:
        print(f"[发送异常] DataNode {datanode_name}：{e}")

def run_distributed_inference_keep_connection(namenode, round_idx):
    """使用已有连接的 NameNode 运行一轮推理"""
    print(f"\n========== 第 {round_idx} 轮运行 ==========")
    
    input_tensor = sample_tensor
    transfer_time = []
    thread_list = []
    thread_start_time = [0] * datanode_num
    thread_end_time = [0] * datanode_num
    thread_time = [[] for _ in range(datanode_num)]

    start_time = time.time()
    middle_output = input_tensor
    thread = [0] * datanode_num
    current_divide_record = None
    
    for layer_it in range(1, total_length + 1):
        print(f"计算第 {layer_it} 层")
        if layer_it > conv_length:
            linear_start = time.time()
            middle_output = inference_model(middle_output, layer_it, layer_it)
            print(f'全连接层耗时：{time.time() - linear_start:.3f}s')
            print("全连接层输出数据大小：", middle_output.size())
        elif layer_it in maxpool_layer:
            pool_start = time.time()
            middle_output = inference_model(middle_output, layer_it, layer_it)
            print(f'池化层耗时：{time.time() - pool_start:.3f}s')
            print("池化层输出数据大小：", middle_output.size())
        else:
            divided_tensor, current_divide_record = tensor_divide_by_computing_and_fill(
                middle_output,
                datanode_num=datanode_num,
                cross_layer=1,
                computing_power=computing_power
            )

            for i in range(datanode_num):
                thread[i] = threading.Thread(
                    target=send_total_data,
                    args=(i, divided_tensor[i], layer_it, layer_it, transfer_time)
                )
                thread_start_time[i] = time.time()
                thread[i].start()

            # 等待线程完成并统计耗时
            for i in range(datanode_num):
                thread[i].join()
                thread_end_time[i] = time.time()
                thread_time[i].append(thread_end_time[i] - thread_start_time[i])

            merged_tensor = namenode.get_merged_total_tensor()
            middle_output = merged_tensor
            print("合并后的 middle_output：", middle_output.size())
    
    total_time = time.time() - start_time
    total_transfer = sum(transfer_time)
    
    print(f"\n第 {round_idx} 轮完成")
    print(f"总耗时：{total_time:.3f}s | 总传输耗时：{total_transfer:.3f}s")

    if round_idx > WARM_UP_ROUNDS:
        print(f"\n第 {i} 轮为有效结果, 输出时间")

        for i in range(datanode_num): print('Thread time: %0.3fs, Thread counts: %d' % (sum(thread_time[i]), len(thread_time[i])))

        for i in range(len(transfer_time)): print("transfer_time: ", transfer_time[i])
        print('NameNode Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))

if __name__ == "__main__":
    # 只初始化一次 NameNode 连接
    print("NameNode 持久化启动，建立所有连接...")
    namenode = Network_init_namenode(namenode_num=namenode_num, datanode_num=datanode_num)
    time.sleep(2)  # 等待 DataNode 就绪
    print("所有连接已建立，开始多轮推理...")
    
    for i in range(1, TOTAL_ROUNDS + 1):
        run_distributed_inference_keep_connection(namenode, i)
        time.sleep(0.5)  # 可选：轮次间短暂休息
    
    # 最后关闭连接
    namenode.close_all()
    print("\n========== 所有轮次完成 ==========")
    print(f"预热轮次：{WARM_UP_ROUNDS} | 有效轮次：{VALID_ROUNDS}")
    print("关闭 NameNode 所有 Socket 连接")