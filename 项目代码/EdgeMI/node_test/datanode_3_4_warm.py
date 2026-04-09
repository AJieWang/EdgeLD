import sys

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up, VGG_model
# from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import divied_middle_output
import torch.nn as nn
import torch, time
import threading

# 初始参数设置
num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
datanode_name = 3
cross_layer = 1

# 加载、初始化模型
inference_model = VGG_model()
# 初始化网络通信模型，参数人为设定
maxpool_layer = inference_model.get_maxpool_layer()
# 循环计算VGG16网络,获得网络人工划分的长度
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()

# Warm-up 配置（与env1保持一致）
WARM_UP_ROUNDS = 3
VALID_ROUNDS = 2
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS


def datanode_persistent():
    """持久化运行的 DataNode，处理多轮推理请求（env4场景）"""
    print(f"\n===== DataNode 0 持久化启动（场景4） =====")

    # 只初始化一次，保持连接
    datanode = Network_init_datanode(
        namenode_num=namenode_num,
        datanode_num=datanode_num,
        datanode_name=datanode_name
    )

    round_idx = 0

    while round_idx < TOTAL_ROUNDS:
        round_idx += 1

        datanode.set_last_inference_layer(0)  # 重置为初始值

        pre_conv = []
        transfer_time = []

        print(f"\n----- DataNode {datanode_name} 第 {round_idx} 轮推理开始 -----")

        try:
            while True:
                # 接受来自 namenode 发送的参数
                start, end, recv_tensor = datanode.datanode_recv_data(pre_conv)
                print("接收来自 NameNode 的数据 recv_tensor：", recv_tensor.size())
                print("要求计算 %d - %d " % (start, end))
                print("datanode.get_last_inference_layer() : ", datanode.get_last_inference_layer())

                # 推理计时开始
                conv_start = time.time()

                # 是否是第一层推理( 通过判断 datanode 设定的初始化 ) 或者是 池化层的后一层
                if (datanode.get_last_inference_layer() == 0) or (start - 1 in maxpool_layer):
                    # recv_tensor为完整的计算tensor，不需要合并或者拆分
                    middle_output = inference_model(recv_tensor, start, end)
                    print("计算第 %d 层的 middle_output:" % start, middle_output.size())
                    # 如果下一层是 maxpool， 需要将数据全部发送至 namenode
                    if end + 1 in maxpool_layer:
                        datanode.datanode_send_data(middle_output, transfer_time, start, end)
                        print("发送第 %d 层的 send_tensor:" % end, middle_output.size())
                    # 设置 计算记录 参数
                    datanode.set_last_inference_layer(end)

                # 推理计时结束
                conv_end = time.time()
                print(f'卷积层{start}-{end} 耗时：{conv_end - conv_start:.3f}s')

                print("完成 %d - %d 的推理任务，并返回计算结果\n" % (start, end))
                if end >= conv_length - 1:
                    print(f"第 {round_idx} 轮推理完成")
                    break

            # 仅在有效轮次输出时间统计（与env1逻辑一致）
            if round_idx > WARM_UP_ROUNDS:
                for i in range(len(pre_conv)):
                    print("pre_conv: ", pre_conv[i])
                print(f'DataNode_{datanode_name} Pre_conv time: %0.3fs, Pre_conv counts: %d' % (sum(pre_conv), len(pre_conv)))

                for i in range(len(transfer_time)):
                    print("transfer_time: ", transfer_time[i])
                print(
                    f'DataNode_{datanode_name} Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))

        except (BrokenPipeError, ConnectionResetError):
            print(f"第 {round_idx} 轮：连接被 NameNode 关闭，等待新连接...")
            # 重新建立连接
            datanode.close()
            time.sleep(1)
            datanode = Network_init_datanode(
                namenode_num=namenode_num,
                datanode_num=datanode_num,
                datanode_name=datanode_name
            )
            continue

        except Exception as e:
            print(f"第 {round_idx} 轮发生错误: {e}")
            break

    # 关闭连接
    datanode.close()
    print(f"关闭 DataNode {datanode_name} 的Socket连接")
    print(f"DataNode{datanode_name} 持久化连接已关闭")


if __name__ == "__main__":
    datanode_persistent()