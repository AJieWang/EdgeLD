import sys

sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up, VGG_model
from VGG.tensor_op import divied_middle_output
import torch.nn as nn
import torch, time
import threading

# 初始参数设置
num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
datanode_name = 2
cross_layer = 1

# 加载、初始化模型
inference_model = VGG_model()
maxpool_layer = inference_model.get_maxpool_layer()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()

WARM_UP_ROUNDS = 3
VALID_ROUNDS = 2
TOTAL_ROUNDS = WARM_UP_ROUNDS + VALID_ROUNDS

def datanode_persistent():
    """持久化运行的 DataNode，处理多轮推理请求"""
    print(f"\n===== DataNode {datanode_name} 持久化启动 =====")
    
    # 只初始化一次，保持连接
    datanode = Network_init_datanode(
        namenode_num=namenode_num,
        datanode_num=datanode_num,
        datanode_name=datanode_name
    )
    
    round_idx = 0

    while round_idx < TOTAL_ROUNDS:
        round_idx += 1

        pre_conv = []
        transfer_time = []

        print(f"\n----- DataNode {datanode_name} 第 {round_idx} 轮推理开始 -----")
        
        try:
            while True:
                start, end, recv_tensor = datanode.datanode_recv_data(pre_conv)
                # recv_tensor = recv_tensor.unsqueeze(0)
                
                # 推理
                conv_start = time.time()
                out = inference_model(recv_tensor, start, end)
                print(f"卷积层{start}-{end} 耗时：{time.time() - conv_start:.3f}s")
                
                # 回传
                datanode.datanode_send_data(out, transfer_time, start, end)
                
                if end >= conv_length - 1:
                    print(f"第 {round_idx} 轮推理完成")
                    break
            if round_idx > WARM_UP_ROUNDS:
                    for i in range(len(pre_conv)): print("pre_conv: ", pre_conv[i])
                    print(f'DataNode_{datanode_name} Pre_conv time: %0.3fs, Pre_conv counts: %d' % (sum(pre_conv), len(pre_conv)))

                    # # for i in range(len(conv_time)): print("conv_time: ", conv_time[i])
                    # print('DataNode_0 Convolution time: %0.3fs, Convolution counts: %d' % (sum(conv_time), len(conv_time)))

                    for i in range(len(transfer_time)): print("transfer_time: ", transfer_time[i])
                    print(f'DataNode_{datanode_name} Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))

        except (BrokenPipeError, ConnectionResetError):
            print(f"第 {round_idx} 轮：连接被 NameNode 关闭，等待新连接...")
            # 重新建立连接
            datanode.close()
            time.sleep(1)
            datanode = Network_init_datanode(
                namenode_num=namenode_num,
                datanode_num=datanode_num,
                datanode_name=0
            )
            continue
            
        except Exception as e:
            print(f"第 {round_idx} 轮发生错误: {e}")
            break
                

    datanode.close()
    print(f"关闭 DataNode {datanode_name} 的Socket连接")
    print("DataNode 持久化连接已关闭")

if __name__ == "__main__":
    datanode_persistent()