import sys
sys.path.append("../..")
sys.path.append("..")

from node_test.network_op import Network_init_datanode, Network_init_namenode
from node_test.num_set_up import Num_set_up
from VGG.mydefine_VGG13 import VGG_model
from VGG.tensor_op import divied_middle_output
import torch.nn as nn
import torch, time
import threading

num_set_up = Num_set_up()
namenode_num = num_set_up.get_namenode_num()
datanode_num = num_set_up.get_datanode_num()
datanode_name = 0
cross_layer = 1

inference_model = VGG_model()
datanode = Network_init_datanode(namenode_num = namenode_num, datanode_num = datanode_num, datanode_name = datanode_name)
maxpool_layer = inference_model.get_maxpool_layer()
conv_length = inference_model.get_conv_length()
total_length = inference_model.get_total_length()

pre_conv = []
transfer_time = []

VGG16_COMPUTE_BLOCKS = {
    1: 3,
    4: 6,
    7: 10,
    11: 14,
    15: 18,
}

VGG13_COMPUTE_BLOCKS = {
    1: 3,
    4: 6,
    7: 9,
    10: 12,
    13: 15,
}

if __name__ == "__main__":

    print("进入计算场景 4 (池化层由DataNode计算版)")
    print("#### 进入模型阶段推理 ####\n")
    while True:
        start, end, recv_tensor = datanode.datanode_recv_data(pre_conv)
        print ("接收来自 NameNode 的数据 recv_tensor：", recv_tensor.size())
        print ("要求计算 %d - %d " % (start, end) )
        
        if start in VGG13_COMPUTE_BLOCKS:
            block_end = VGG13_COMPUTE_BLOCKS[start]
            print(f"计算卷积块: 层 {start} - {block_end} (包含池化层)")
            middle_output = inference_model(recv_tensor, start, block_end)
            print ("计算完成 middle_output:" , middle_output.size())
            datanode.datanode_send_data(middle_output, transfer_time, start, block_end)
            print(f"发送第 {block_end} 层的结果 (包含池化)")
        
        print("完成 %d - %d 的推理任务，并返回计算结果\n" % (start, end))
        if end >= conv_length - 1:
            print("DataNode %d 结束推理")
            break
    
    for i in range(len(transfer_time)): print("transfer_time: ", transfer_time[i])
    print('DataNode_0 Transfer time: %0.3fs, Transfer counts: %d' % (sum(transfer_time), len(transfer_time)))

    time.sleep(2)
    datanode.close()
    print("关闭 DataNode %d 的Socket连接" % datanode_name)
