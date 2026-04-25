import sys
sys.path.append("../..")
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# 针对设备计算能力 和 网络通信状态的
def tensor_divide_by_computing_network_and_fill(original_tensor, datanode_num = 1, cross_layer = 1, in_chanel = 1,
                                                out_chanel = 1, kernerl_size = 3, computing_power = [], network_state = []):
    a = 0



# 定义一个tensor直接分割, original_tensor默认为4维, 划分后的[start, end],含start，不包含end
def tensor_divide(original_tensor, divide_num = 1):
    divided_tensor = []
    divide_record = np.zeros((divide_num, 2), dtype=np.int)
    if divide_num == 1:
        return original_tensor, divide_record
    else:
        _, _, _, width = original_tensor.size()
        # [0]start,[1]end
        for it in range(divide_num):
            start = int(width / divide_num * it)
            end =  int( width / divide_num * (it + 1) )
            print("[ %d, %d]" % (start, end))
            # 记录
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 根据start得到tensor
            temp_tensor = original_tensor[:, :, :, start:end]
            # 放入list
            divided_tensor.append(temp_tensor)
    # 返回最终的结果
    return divided_tensor, divide_record

# 定义一个tensor直接分割, original_tensor默认为4维, 划分后的[start, end],含start，不包含end
def tensor_divide_and_fill(original_tensor, datanode_num = 1, cross_layer = 1):
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        _, _, _, width = original_tensor.size()
        # [0]start,[1]end
        for it in range(datanode_num):
            start = int(width / datanode_num * it)
            end =  int( width / datanode_num * (it + 1) )
            print ("[ %d, %d]" %(start, end))
            # 记录
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start:int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
    # 返回最终的结果
    return divided_tensor, divide_record


# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_fill(original_tensor, datanode_num = 1, cross_layer = 1, computing_power=[]):
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        _, _, _, width = original_tensor.size()
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num+1):
            sum_computing_power.append(sum(computing_power[0:i]))
        # [0]start,[1]end
        for it in range(datanode_num):
            start = int(sum_computing_power[it]/total_computing_power * width)
            end = int(sum_computing_power[it+1]/total_computing_power * width)
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start:int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer):int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
    # 返回最终的结果
    return divided_tensor, divide_record


# #############################################################################################################
# 时间估计
def get_prediction_time(datanode_num = 0, index = 0, length = 0, cross_layer = 1, computing_a = [],
                        computing_b = [], network_state = [], input_param = [], c_out = 0):
    input_number, c_in, height, width = input_param
    if c_out == 0:
        c_out = c_in
    else:
        c_out = c_out
    kernel = 3
    # 计算 FLOPs
    FLOPs = input_number * 2 * height * length * c_out * (kernel * kernel * c_in + 1)
    # 计算时间
    comp_time = computing_a[index] * FLOPs + computing_b[index]
    # 通信开销
    comm_data = input_number * c_in * height * 4.0 / network_state[index]
    comm_time = 0
    # 判断是否是边界
    if index == 0 or index == datanode_num - 1:
        comm_time = comm_data * (cross_layer + 2 * length)
    # 中间情况
    else:
        comm_time = 2 * comm_data * (cross_layer + length)
    prediction_time = comp_time + comm_time
    return prediction_time



# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network(original_tensor, datanode_num = 1, cross_layer = 1,
                                        computing_power = [], computing_a = [], computing_b = [], network_state = [], c_out = 0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0 : i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it+1]/total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num = datanode_num, index = it, length = length[it], cross_layer = cross_layer, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while(True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if ( diff < 0.02 or min(length) <= 1):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num = datanode_num, index = index_max, length = length[index_max], cross_layer = 1, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num = datanode_num, index = index_min, length = length[index_min], cross_layer = 1, computing_a = computing_a,
                        computing_b = computing_b, network_state = network_state, input_param=input_param, c_out = c_out)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)
        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        for it in range(datanode_num):
            end = start + length[it]
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start : int(end + cross_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - cross_layer) : int(end + cross_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    return divided_tensor, divide_record

# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network_pooled(original_tensor, datanode_num=1, cross_layer=1,
                                           computing_power=[], computing_a=[], computing_b=[], network_state=[],
                                           c_out=0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0: i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it + 1] / total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num=datanode_num, index=it, length=length[it],
                                                      cross_layer=cross_layer, computing_a=computing_a,
                                                      computing_b=computing_b, network_state=network_state,
                                                      input_param=input_param, c_out=c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while (True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if (diff < 0.02 or min(length) <= 1):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num=datanode_num, index=index_max,
                                                             length=length[index_max], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num=datanode_num, index=index_min,
                                                             length=length[index_min], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)

        # ===================== 新增优化：保证每个切分width为偶数 =====================
        # 调整length数组，确保每个元素都是偶数，且总和不变
        total_length = sum(length)
        # 遍历调整每个长度为偶数
        for i in range(datanode_num):
            if length[i] % 2 != 0:
                length[i] -= 1  # 奇数减1变为偶数，保证最小幅度调整
        # 补偿因减1损失的长度，保证总宽度不变（偶数补偿，不破坏偶数性）
        compensate = total_length - sum(length)
        idx = 0
        while compensate > 0:
            length[idx] += 2
            compensate -= 2
            idx = (idx + 1) % datanode_num
        # ==========================================================================

        # # 已经得到length，根据length确定划分范围
        # start = 0
        # end = 0
        # for it in range(datanode_num):
        #     end = start + length[it]
        #     # print("[ %d, %d]" % (start, end))
        #     divide_record[it][0] = start
        #     divide_record[it][1] = end
        #     # 判断划分的位置
        #     temp_tensor = 0
        #     if it == 0:
        #         # 最左边划分
        #         temp_tensor = original_tensor[:, :, :, start: int(end + cross_layer)]
        #     elif it == datanode_num - 1:
        #         # 最右边划分
        #         temp_tensor = original_tensor[:, :, :, int(start - cross_layer): end]
        #     else:
        #         # 中间非边界情况
        #         temp_tensor = original_tensor[:, :, :, int(start - cross_layer): int(end + cross_layer)]
        #     # 放入list
        #     divided_tensor.append(temp_tensor)
        #     # 更换起始位置。
        #     start = end

        # ======================================= Fix:张量（Tensor）的分段 + 边界扩展填充 =======================================
        start = 0
        for it in range(datanode_num):
            end = start + length[it]
            divide_record[it][0] = start
            divide_record[it][1] = end

            # 计算本段左右各需扩展多少（边界段只向内扩展）
            left_pad = cross_layer if it > 0 else 0
            right_pad = cross_layer if it < datanode_num - 1 else 0

            left_idx = start - left_pad
            right_idx = end + right_pad

            # 取出与原始张量有交集的部分
            orig_len = original_tensor.shape[-1]
            valid_left = max(left_idx, 0)
            valid_right = min(right_idx, orig_len)

            temp_tensor = original_tensor[..., valid_left:valid_right]

            # 计算需要在左右填充的零的数量
            pad_left = valid_left - left_idx
            pad_right = right_idx - valid_right

            if pad_left > 0 or pad_right > 0:
                # F.pad 对最后一维的填充顺序为 (left, right)
                temp_tensor = F.pad(temp_tensor, (pad_left, pad_right), mode='constant', value=0)

            divided_tensor.append(temp_tensor)
            start = end
        # ======================================= Fix:张量（Tensor）的分段 + 边界扩展填充 =======================================
    # 返回最终的结果
    return divided_tensor, divide_record

# #############################################################################################################
# 根据计算能力划分区域, original_tensor默认为4维，划分后的[start, end],含start，不包含end
def tensor_divide_by_computing_and_network_pabc(original_tensor, datanode_num=1, cross_layer=1,
                                           computing_power=[], computing_a=[], computing_b=[], network_state=[],
                                           c_out=0):
    # 优化步长
    step = 1
    divided_tensor = []
    divide_record = np.zeros((datanode_num, 2), dtype=np.int)
    input_param = []
    if datanode_num == 1:
        return original_tensor, divide_record
    else:
        input_number, c_in, height, width = original_tensor.size()
        input_param.append(input_number)
        input_param.append(c_in)
        input_param.append(height)
        input_param.append(width)
        # 提前计算求和
        total_computing_power = 0
        for i in range(datanode_num):
            total_computing_power += computing_power[i]
        sum_computing_power = []
        for i in range(datanode_num + 1):
            sum_computing_power.append(sum(computing_power[0: i]))

        # 定义划分长度
        length = []
        # 时间开销
        prediction_time = []
        for i in range(datanode_num):
            length.append(0)
            prediction_time.append(0)
        for it in range(datanode_num):
            length[it] = int(sum_computing_power[it + 1] / total_computing_power * width) - \
                         int(sum_computing_power[it] / total_computing_power * width)
        for it in range(datanode_num):
            prediction_time[it] = get_prediction_time(datanode_num=datanode_num, index=it, length=length[it],
                                                      cross_layer=cross_layer, computing_a=computing_a,
                                                      computing_b=computing_b, network_state=network_state,
                                                      input_param=input_param, c_out=c_out)
        iter = 0
        iter_stop = 30
        diff = 0
        # 判断退出条件,1、max与min差值小于10ms，或者差值变化很小，或者某一个i对应的长度接近 1
        while (True):
            iter += 1
            # 判断是否到轮次上限
            if iter == iter_stop:
                break
            # 找出时间最值及下标
            max_value = max(prediction_time)
            min_value = min(prediction_time)
            index_max = prediction_time.index(max_value)
            index_min = prediction_time.index(min_value)
            last_diff = diff
            diff = max_value - min_value
            # 判断退出条件
            if (diff < 0.02 or min(length) <= 1):
                break
            last_diff = diff
            length[index_max] -= step
            length[index_min] += step
            # 出错
            prediction_time[index_max] = get_prediction_time(datanode_num=datanode_num, index=index_max,
                                                             length=length[index_max], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
            prediction_time[index_min] = get_prediction_time(datanode_num=datanode_num, index=index_min,
                                                             length=length[index_min], cross_layer=1,
                                                             computing_a=computing_a,
                                                             computing_b=computing_b, network_state=network_state,
                                                             input_param=input_param, c_out=c_out)
        #     print (length)
        #     print (prediction_time)
        # print(length)
        # print(prediction_time)

        # ===================== 优化：保证每个切分 width 为 4 的倍数 =====================
        total_length = sum(length)
        # 第一步：把每个长度向下取为 4 的倍数
        for i in range(datanode_num):
            length[i] = (length[i] // 4) * 4  # 向下取 4 的倍数

        # 第二步：补偿总长度，保证总和不变，且每个块仍然是 4 的倍数
        compensate = total_length - sum(length)
        idx = 0
        while compensate > 0:
            length[idx] += 4  # 每次 +4，保持是 4 的倍数
            compensate -= 4
            idx = (idx + 1) % datanode_num
        # ==========================================================================

        # 已经得到length，根据length确定划分范围
        start = 0
        end = 0
        divide_layer = cross_layer + 2
        for it in range(datanode_num):
            end = start + length[it]
            # print("[ %d, %d]" % (start, end))
            divide_record[it][0] = start
            divide_record[it][1] = end
            # 判断划分的位置
            temp_tensor = 0
            if it == 0:
                # 最左边划分
                temp_tensor = original_tensor[:, :, :, start: int(end + divide_layer)]
            elif it == datanode_num - 1:
                # 最右边划分
                temp_tensor = original_tensor[:, :, :, int(start - divide_layer): end]
            else:
                # 中间非边界情况
                temp_tensor = original_tensor[:, :, :, int(start - divide_layer): int(end + divide_layer)]
            # 放入list
            divided_tensor.append(temp_tensor)
            # 更换起始位置。
            start = end
    # 返回最终的结果
    return divided_tensor, divide_record

# 定义一个推理后的tensor，拆除无关部分
def merge_total_tensor(divided_tensor = [], divide_record = [], cross_layer = 1):
    '''
    :param divided_tensor: 需要合并的tensor
    :param divide_record:  之前拆分位置的记录
    :return: 合并后的tensor
    '''
    length = len(divided_tensor)
    if length == 0:
        return 0
    if length == 1:
        return divided_tensor[0][:, :, :, :]
    merged_tensor = 0
    for it in range(length):
        if it == 0:
            # 最左侧
            merged_tensor = divided_tensor[it][:, :, :, 0:-cross_layer]
        elif it == length -1:
            # 最右侧
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, cross_layer:]), 3)
        else:
            # 中间非边界
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, cross_layer: -cross_layer]), 3)
    return merged_tensor

# 定义一个推理后的tensor，拆除无关部分
def merge_total_tensor_pooled(divided_tensor = [], divide_record = [], cross_layer = 1):
    '''
    :param divided_tensor: 需要合并的tensor
    :param divide_record:  之前拆分位置的记录
    :return: 合并后的tensor
    '''
    length = len(divided_tensor)
    if length == 0:
        return 0
    if length == 1:
        return divided_tensor[0][:, :, :, :]
    merged_tensor = 0
    # ===================== 新增优化：切分合并保证大小 ============================
    divide_layer = cross_layer - 1
    # ===========================================================================
    for it in range(length):
        if it == 0:
            # 最左侧
            merged_tensor = divided_tensor[it][:, :, :, 0:-divide_layer]
        elif it == length -1:
            # 最右侧
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, divide_layer:]), 3)
        else:
            # 中间非边界
            merged_tensor = torch.cat((merged_tensor, divided_tensor[it][:, :, :, divide_layer: -divide_layer]), 3)
    return merged_tensor

# 定义一个推理后的tensor，拆除无关部分
def merge_total_tensor_pabc(divided_tensor=[], divide_record=[], cross_layer=1):
    '''
    单纯拼接 divided_tensor 中的所有张量，在第4维（dim=3）合并
    :param divided_tensor: 需要合并的 tensor 列表
    :param divide_record:  无用参数（保留兼容）
    :return: 拼接后的完整 tensor
    '''
    length = len(divided_tensor)
    if length == 0:
        return 0

    # 直接在 dim=3 拼接所有张量，不做任何裁剪/切片
    merged_tensor = torch.cat(divided_tensor, dim=3)
    return merged_tensor

# 聚合 差分传输 的tensor，暂时不需要写
def merge_part_tensor(divided_tensor = [], divide_record = [], cross_layer = 1):
    return 0

# 用于datanode的差值交换，拆开计算结果tensor，分为  saved_tensor, divied_tensor
def divied_middle_output(input_tensor = 0, datanode_num = 1, datanode_name = 0, cross_layer = 1):
    saved_tensor = torch.rand(1, 1, 1, 1)
    divied_tensor = []
    # 最左
    if datanode_name == 0 :
        saved_tensor = input_tensor[:, :, :, 0 : -cross_layer]
        divied_tensor.append(  input_tensor[:, :, :, -cross_layer : ]  )
    # 最右
    elif datanode_name == datanode_num - 1:
        saved_tensor = input_tensor[:, :, :, cross_layer : ]
        divied_tensor.append(  input_tensor[:, :, :, 0 : cross_layer ]   )
    # 中间
    else:
        saved_tensor = input_tensor[:, :, :, cross_layer : -cross_layer]
        divied_tensor.append(  input_tensor[:, :, :, 0 : cross_layer]  )
        divied_tensor.append(  input_tensor[:, :, :, -cross_layer : ])
    return saved_tensor, divied_tensor


# 输入一个tensor，得到这个tensor的比特长度，传输数据时统计
def get_tensor_bytes_length( input_tensor ):
    input_num, input_chanel, height, width = input_tensor.size()
    numbers = input_num * input_chanel * height * width
    bytes_length = int(numbers * 4)
    return bytes_length

# 统计卷积运算的 FLOPS
def get_conv_tensor_flops(in_chanel = 1, out_chanel = 1, kernel_size = 3, input_height = 1, input_width = 1):
    return 2 * input_height * input_width * out_chanel * (in_chanel * kernel_size * kernel_size + 1)
# 统计全连接层的 FLOPS
def get_fully_tensor_flops(input = 1, output = 1):
    return output * (2 * input - 1)

# 主函数
# if __name__ == "__main__":
#     width = 224
#     num = 3
#     VGG16 = VGG_model()
#     input = torch.rand(1, 3, width, width)
#     computing_power = [4, 1, 4, 8, 7, 4, 4]
#     temp = [1, 2, 4, 5, 6]
#
#     divided_tensor, divide_record = tensor_divide_by_computing_and_fill(input, num, cross_layer = 2, computing_power = computing_power)
#     # 测试卷积后是否相同
#     output_1 = VGG16(input, 1, 2)
#     print (output_1.size())
#     output_tensor = []
#     for i in range(num):
#         output_tensor.append(VGG16(divided_tensor[i], 1, 2))
#     print ( len(output_tensor) )
#     merged_tensor = merge_total_tensor(output_tensor, divide_record, cross_layer = 2)
#     print(torch.equal(output_1, merged_tensor))

