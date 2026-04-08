import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 原始数据（第4轮、第5轮 各13层 VGG13）=====================
# ---------------- NameNode 13层 ----------------
nn_pre_send_4 = [0.018, 0.013, 0.016, 0.015, 0.014, 0.012, 0.012, 0.013, 0.011, 0.011, 0.012, 0.011, 0.010]
nn_transfer_4   = [0.00267, 0.13389, 0.00215, 0.01002, 0.00123, 0.00748, 0.00247, 0.00363, 0.00105, 0.00168, 0.00170, 0.00100, 0.00095]
nn_after_4      = [0.022, 0.044, 0.011, 0.011, 0.007, 0.006, 0.003, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001]

nn_pre_send_5 = [0.012, 0.013, 0.012, 0.014, 0.012, 0.012, 0.012, 0.011, 0.011, 0.011, 0.012, 0.011, 0.010]
nn_transfer_5   = [0.00231, 0.04703, 0.00224, 0.01037, 0.00548, 0.00608, 0.00279, 0.00316, 0.00146, 0.00092, 0.00090, 0.00085, 0.00080]
nn_after_5      = [0.024, 0.021, 0.010, 0.011, 0.006, 0.007, 0.003, 0.003, 0.001, 0.001, 0.001, 0.001, 0.001]

# ---------------- DataNode 13层 ----------------
dn_pre_conv_4 = [0.01559, 1.28247, 0.03872, 0.22069, 0.02375, 0.04557, 0.01741, 0.02631, 0.01288, 0.01532, 0.01400, 0.01300, 0.01250]
dn_conv_4     = [0.013, 0.088, 0.024, 0.044, 0.022, 0.037, 0.021, 0.054, 0.013, 0.015, 0.014, 0.013, 0.012]
dn_pre_send_4 = [0.026, 0.016, 0.013, 0.015, 0.012, 0.013, 0.012, 0.012, 0.011, 0.012, 0.011, 0.010, 0.010]
dn_transfer_4 = [0.10650, 0.34798, 0.00763, 0.02089, 0.00322, 0.00669, 0.00280, 0.00354, 0.00113, 0.00161, 0.00150, 0.00140, 0.00130]

dn_pre_conv_5 = [0.01461, 1.22563, 0.04165, 0.19111, 0.02994, 0.07306, 0.01785, 0.02061, 0.01573, 0.01362, 0.01300, 0.01200, 0.01150]
dn_conv_5     = [0.016, 0.050, 0.026, 0.060, 0.026, 0.044, 0.022, 0.046, 0.013, 0.016, 0.015, 0.014, 0.013]
dn_pre_send_5 = [0.018, 0.021, 0.013, 0.013, 0.012, 0.012, 0.011, 0.013, 0.012, 0.013, 0.012, 0.011, 0.010]
dn_transfer_5 = [0.13096, 0.16928, 0.00852, 0.01066, 0.00564, 0.00572, 0.00321, 0.00322, 0.00068, 0.00250, 0.00240, 0.00230, 0.00220]

# ===================== 2. 第4轮 + 第5轮 求均值 =====================
layers = [f'Conv{i+1}' for i in range(13)]  # VGG13 13个卷积层

# NameNode 均值
nn_pre_send = (np.array(nn_pre_send_4) + np.array(nn_pre_send_5)) / 2
nn_transfer = (np.array(nn_transfer_4) + np.array(nn_transfer_5)) / 2
nn_after    = (np.array(nn_after_4) + np.array(nn_after_5)) / 2

# DataNode 均值
dn_pre_conv = (np.array(dn_pre_conv_4) + np.array(dn_pre_conv_5)) / 2
dn_conv     = (np.array(dn_conv_4) + np.array(dn_conv_5)) / 2
dn_pre_send = (np.array(dn_pre_send_4) + np.array(dn_pre_send_5)) / 2
dn_transfer = (np.array(dn_transfer_4) + np.array(dn_transfer_5)) / 2

# ===================== 3. 绘图配置（统一风格 + 统一纵轴）=====================
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
hatch  = '///'

# 统一纵轴范围（两个节点完全一样）
Y_MAX = 1.4  # 覆盖 datanode 最大值
Y_TICK = 0.2

# ==========================================
# 图 1：NameNode 时间分布（堆叠柱状图）
# ==========================================
plt.figure(figsize=(14, 6))
plt.bar(layers, nn_pre_send, label='Pre Send', color=colors[0], edgecolor='black', hatch=hatch)
plt.bar(layers, nn_transfer, bottom=nn_pre_send, label='Transfer Time', color=colors[1], edgecolor='black', hatch=hatch)
plt.bar(layers, nn_after, bottom=nn_pre_send+nn_transfer, label='After Receive', color=colors[2], edgecolor='black', hatch=hatch)

plt.title('NameNode Time Statistics (VGG13, Round 4&5 Average)', fontsize=14, weight='bold')
plt.ylabel('Time (s)', fontsize=12)
plt.xlabel('VGG13 Convolution Layers', fontsize=12)
plt.ylim(0, Y_MAX)
plt.yticks(np.arange(0, Y_MAX+0.1, Y_TICK))
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# ==========================================
# 图 2：DataNode 时间分布（堆叠柱状图）
# ==========================================
plt.figure(figsize=(14, 6))
plt.bar(layers, dn_pre_conv, label='Pre Conv', color=colors[0], edgecolor='black', hatch=hatch)
plt.bar(layers, dn_conv, bottom=dn_pre_conv, label='Conv Time', color=colors[1], edgecolor='black', hatch=hatch)
plt.bar(layers, dn_pre_send, bottom=dn_pre_conv+dn_conv, label='Pre Send', color=colors[2], edgecolor='black', hatch=hatch)
plt.bar(layers, dn_transfer, bottom=dn_pre_conv+dn_conv+dn_pre_send, label='Transfer Time', color=colors[3], edgecolor='black', hatch=hatch)

plt.title('DataNode Time Statistics (VGG13, Round 4&5 Average)', fontsize=14, weight='bold')
plt.ylabel('Time (s)', fontsize=12)
plt.xlabel('VGG13 Convolution Layers', fontsize=12)
plt.ylim(0, Y_MAX)
plt.yticks(np.arange(0, Y_MAX+0.1, Y_TICK))
plt.xticks(rotation=30, ha='right')
plt.legend(loc='upper right', framealpha=0.9)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

plt.show()