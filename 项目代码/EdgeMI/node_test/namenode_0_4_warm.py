import sys

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up
import torch
import threading
import time
import torch.nn as nn
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import tensor_divide, tensor_divide_and_fill, tensor_divide_by_computing_and_fill, \
    tensor_divide_by_computing_network_and_fill, tensor_divide_by_computing_and_network
from VGG.tensor_op import merge_total_tensor, merge_part_tensor
from network_and_computing.network_and_computing_record import Network_And_Computing

# ====================== 新增 Warm Up 配置 ======================
WARM_UP_ROUNDS = 3  # 预热轮次
VALID_ROUNDS = 2  # 有效轮次
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS
# ==============================================================

# 适用于场景4：各设备算力/通信不同、多层差异数据交换、池化层全量交换
num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()

# 获得 datanode 设备的 计算能力 和 通信能力
network_and_computing = Network_And_Computing()
computing_power = network_and_computing.get_computing_power(datanode_num)
network_state = network_and_computing.get_network_state(datanode_num)
computing_a = network_and_computing.get_computing_a(datanode_num)
computing_b = network_and_computing.get_computing_b(datanode_num)

# 初始化，加载网络
inference_model = VGG_model()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()
c_out_list = inference_model.get_c_out()
maxpool_layer = inference_model.get_maxpool_layer()
width = 224

# ====================== 时间统计相关全局变量 ======================
transfer_time = []
thread_start_time = []
thread_end_time = []
thread_time = []


# =================================================================

def send_total_data(datanode_name, input_tensor, start, end, transfer_time_):
    """发送全量数据，带传输时间统计"""
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


def send_part_data(datanode_name, input_tensor, start, end):
    """发送部分数据（env4 中未实际使用，保留原逻辑）"""
    namenode.namenode_send_data(
        datanode_name=datanode_name,
        input_tensor=input_tensor,
        start=start,
        end=end
    )


def get_end_layer(start=1, maxpool_layer=[]):
    """获取当前起始层到下一个池化层的结束层"""
    max_value = max(maxpool_layer) if maxpool_layer else 0
    if start > max_value or start < 1:
        return 0
    for layer in maxpool_layer:
        if layer > start:
            return layer
    return max_value


def run_distributed_inference_env4(namenode, round_idx):
    """运行单轮 env4 推理（带完整时间统计）"""
    print(f"\n========== 第 {round_idx} 轮运行 (Env4) ==========")

    # 重置本轮时间统计
    global transfer_time, thread_start_time, thread_end_time, thread_time
    transfer_time = []
    thread_start_time = [0] * datanode_num
    thread_end_time = [0] * datanode_num
    thread_time = [[] for _ in range(datanode_num)]

    # 初始化输入和线程/接收列表
    input_tensor = torch.rand(1, 3, width, width)
    middle_output = input_tensor
    final_output = torch.rand(1, 100)
    thread = [0] * datanode_num
    recv_tensor_list = [0] * datanode_num

    # 本轮推理开始时间
    round_start_time = time.time()

    if datanode_num != 1:
        for layer_it in range(1, total_length + 1, 1):
            print(f"计算第 {layer_it} 层")
            if layer_it > conv_length:
                # 全连接层计算
                linear_start = time.time()
                middle_output = inference_model(middle_output, layer_it, layer_it)
                linear_end = time.time()
                print(f'全连接层耗时：{linear_end - linear_start:.3f}s')
                print("全连接层输出数据大小：", middle_output.size())
            elif layer_it in maxpool_layer:
                # 池化层计算
                pool_start = time.time()
                middle_output = inference_model(middle_output, layer_it, layer_it)
                pool_end = time.time()
                print(f'池化层耗时：{pool_end - pool_start:.3f}s')
                print("池化层输出数据大小：", middle_output.size())
            elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
                # 第一层/池化层后一层：全量传输
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print(f"第一层/池化层后一层，跨层数：{cross_layer}")

                # 张量划分
                divided_tensor, divide_record = tensor_divide_by_computing_and_network(
                    middle_output,
                    datanode_num=datanode_num,
                    cross_layer=cross_layer,
                    computing_power=computing_power,
                    computing_a=computing_a,
                    computing_b=computing_b,
                    network_state=network_state,
                    c_out=c_out_list[layer_it]
                )

                # 多线程发送数据
                for i in range(datanode_num):
                    thread[i] = threading.Thread(
                        target=send_total_data,
                        args=(i, divided_tensor[i], start, end, transfer_time)
                    )
                    thread_start_time[i] = time.time()
                    thread[i].start()

                # 等待线程完成并统计耗时
                for i in range(datanode_num):
                    thread[i].join()
                    thread_end_time[i] = time.time()
                    thread_time[i].append(thread_end_time[i] - thread_start_time[i])

                # 合并结果
                # temp = namenode.get_recv_tensor_list()
                middle_output = namenode.get_merged_total_tensor(cross_layer=cross_layer)
                print("合并后的 middle_output：", middle_output.size())
            else:
                print(f"NameNode不参与第 {layer_it} 层计算")
                continue
    else:
        # 单 datanode 分支逻辑
        for layer_it in range(1, total_length + 1, 1):
            print(f"计算第 {layer_it} 层")
            if layer_it > conv_length:
                linear_start = time.time()
                middle_output = inference_model(middle_output, layer_it, layer_it)
                print(f'全连接层耗时：{time.time() - linear_start:.3f}s')
                print("全连接层输出数据大小：", middle_output.size())
            elif layer_it in maxpool_layer:
                pool_start = time.time()
                middle_output = inference_model(middle_output, layer_it, layer_it)
                pool_end = time.time()
                print(f'池化层耗时：{pool_end - pool_start:.3f}s')
                print("池化层输出数据大小：", middle_output.size())
            elif (layer_it == 1) or (layer_it - 1 in maxpool_layer):
                start = layer_it
                end = get_end_layer(start, maxpool_layer) - 1
                cross_layer = end - start + 1
                print(f"第一层/池化层后一层，跨层数：{cross_layer}")

                # 多线程发送数据
                for i in range(datanode_num):
                    thread[i] = threading.Thread(
                        target=send_total_data,
                        args=(i, middle_output, start, end, transfer_time)
                    )
                    thread_start_time[i] = time.time()
                    thread[i].start()

                # 等待线程完成并统计耗时
                for i in range(datanode_num):
                    thread[i].join()
                    thread_end_time[i] = time.time()
                    thread_time[i].append(thread_end_time[i] - thread_start_time[i])

                # 合并结果
                middle_output = namenode.get_merged_total_tensor(cross_layer=cross_layer)
                print("合并后的 middle_output：", middle_output.size())
            else:
                print(f"NameNode不参与第 {layer_it} 层计算")
                continue

    # 本轮总耗时统计
    round_total_time = time.time() - round_start_time
    round_total_transfer = sum(transfer_time)

    print(f"\n第 {round_idx} 轮完成 (Env4)")
    print(f"本轮总耗时：{round_total_time:.3f}s | 本轮总传输耗时：{round_total_transfer:.3f}s")

    # 有效轮次打印详细统计（参考 env1 逻辑）
    if round_idx > WARM_UP_ROUNDS:
        print(f"\n第 {i} 轮为有效结果, 输出时间")

        for i in range(datanode_num): print('Thread time: %0.3fs, Thread counts: %d' % (sum(thread_time[i]), len(thread_time[i])))

        for i in range(len(transfer_time)): print("transfer_time: ", transfer_time[i])
        print('NameNode Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))


if __name__ == "__main__":
    # 初始化 NameNode 连接（仅初始化一次）
    print("NameNode Env4 持久化启动，建立所有连接...")
    namenode = Network_init_namenode(namenode_num=namenode_num, datanode_num=datanode_num)
    time.sleep(2)  # 等待 DataNode 就绪
    print("所有连接已建立，开始多轮推理 (Env4)...")

    # 多轮推理（包含预热+有效轮次）
    for round_idx in range(1, TOTAL_ROUNDS + 1):
        run_distributed_inference_env4(namenode, round_idx)
        time.sleep(0.5)  # 轮次间短暂休息

    # 关闭连接
    namenode.close_all()
    print("\n========== 所有轮次完成 ==========")
    print(f"预热轮次：{WARM_UP_ROUNDS} | 有效轮次：{VALID_ROUNDS}")
    print("关闭 NameNode 所有 Socket 连接")