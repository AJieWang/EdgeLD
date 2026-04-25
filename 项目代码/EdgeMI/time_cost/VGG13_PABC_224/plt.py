import matplotlib.pyplot as plt
import numpy as np

from time_data import data_round4, data_round5

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

STAGE_COLORS = {
    'namenode pre_send': '#1f77b4',
    'namenode transfer_time': '#9467bd',
    'datanode pre_conv': '#ff7f0e',
    'datanode conv_time': '#2ca02c',
    'datanode pre_send': '#d62728',
    'datanode transfer_time': '#9467bd',
    'namenode after_receive': '#8c564b',
    'namenode fullyconnect': '#7f7f7f'
}

def calc_mean(val1, val2):
    if isinstance(val1, list):
        return [(val1 + val2) / 2 for val1, val2 in zip(val1, val2)]
    else:
        return (val1 + val2) / 2

avg_data = {key: calc_mean(data_round4[key], data_round5[key]) for key in data_round4}

# ==========================================
# 数据重组：构建 FL 和 Conv 结构
# ==========================================
fl_labels = ['FL1', 'FL2', 'FL3']
fl_values = avg_data['namenode_fullyconnect']
conv_labels = ['Conv1', 'Conv2', 'Conv4', 'Conv5', 'Conv7', 'Conv8', 'Conv10', 'Conv11', 'Conv13', 'Conv14']

stage_names_odd = ['namenode pre_send', 'namenode transfer_time', 'datanode pre_conv', 'datanode conv_time']
stage_names_even = ['datanode pre_send', 'datanode transfer_time', 'namenode after_receive']

conv_stages_odd = {stage: [0] * 10 for stage in stage_names_odd}
conv_stages_even = {stage: [0] * 10 for stage in stage_names_even}

# 数据索引映射:
# i=0(Conv1) -> idx0, i=1(Conv2) -> idx0(空), i=2(Conv4) -> idx1(空), i=3(Conv5) -> idx1
# i=4(Conv7) -> idx2, i=5(Conv8) -> idx2, i=6(Conv10) -> idx3, i=7(Conv11) -> idx3
# i=8(Conv13) -> idx3, i=9(Conv14) -> idx3

# Conv1, Conv7, Conv10, Conv13 (namenode数据)
conv_stages_odd['namenode pre_send'][0] = avg_data['namenode_pre_send'][0]
conv_stages_odd['namenode transfer_time'][0] = avg_data['namenode_transfer_time'][0]
conv_stages_odd['datanode pre_conv'][0] = avg_data['datanode_pre_conv_pool'][0]
conv_stages_odd['datanode conv_time'][0] = avg_data['datanode_conv_pool_time'][0]

conv_stages_odd['namenode pre_send'][4] = avg_data['namenode_pre_send'][1]
conv_stages_odd['namenode transfer_time'][4] = avg_data['namenode_transfer_time'][1]
conv_stages_odd['datanode pre_conv'][4] = avg_data['datanode_pre_conv_pool'][1]
conv_stages_odd['datanode conv_time'][4] = avg_data['datanode_conv_pool_time'][1]

conv_stages_odd['namenode pre_send'][6] = avg_data['namenode_pre_send'][2]
conv_stages_odd['namenode transfer_time'][6] = avg_data['namenode_transfer_time'][2]
conv_stages_odd['datanode pre_conv'][6] = avg_data['datanode_pre_conv_pool'][2]
conv_stages_odd['datanode conv_time'][6] = avg_data['datanode_conv_pool_time'][2]

conv_stages_odd['namenode pre_send'][8] = avg_data['namenode_pre_send'][3]
conv_stages_odd['namenode transfer_time'][8] = avg_data['namenode_transfer_time'][3]
conv_stages_odd['datanode pre_conv'][8] = avg_data['datanode_pre_conv_pool'][3]
conv_stages_odd['datanode conv_time'][8] = avg_data['datanode_conv_pool_time'][3]

# Conv5, Conv8, Conv11, Conv14 (datanode数据)
conv_stages_even['datanode pre_send'][3] = avg_data['datanode_pre_send'][0]
conv_stages_even['datanode transfer_time'][3] = avg_data['datanode_transfer_time'][0]
conv_stages_even['namenode after_receive'][3] = avg_data['namenode_after_receive'][0]

conv_stages_even['datanode pre_send'][5] = avg_data['datanode_pre_send'][1]
conv_stages_even['datanode transfer_time'][5] = avg_data['datanode_transfer_time'][1]
conv_stages_even['namenode after_receive'][5] = avg_data['namenode_after_receive'][1]

conv_stages_even['datanode pre_send'][7] = avg_data['datanode_pre_send'][2]
conv_stages_even['datanode transfer_time'][7] = avg_data['datanode_transfer_time'][2]
conv_stages_even['namenode after_receive'][7] = avg_data['namenode_after_receive'][2]

conv_stages_even['datanode pre_send'][9] = avg_data['datanode_pre_send'][3]
conv_stages_even['datanode transfer_time'][9] = avg_data['datanode_transfer_time'][3]
conv_stages_even['namenode after_receive'][9] = avg_data['namenode_after_receive'][3]

# ==========================================
# 生成柱状图
# ==========================================
plt.figure(figsize=(20, 8))
bar_width = 0.6
x_conv = np.arange(len(conv_labels))

# 绘制奇数位置柱状图 (namenode数据)
bottom_odd = np.zeros(len(conv_labels))
for idx, stage in enumerate(stage_names_odd):
    stage_data = conv_stages_odd[stage]
    plt.bar(x_conv, stage_data, bar_width, bottom=bottom_odd, label=stage, color=STAGE_COLORS[stage], edgecolor='black', alpha=0.8)
    bottom_odd += np.array(stage_data)

# 绘制偶数位置柱状图 (datanode数据)
bottom_even = np.zeros(len(conv_labels))
for idx, stage in enumerate(stage_names_even):
    stage_data = conv_stages_even[stage]
    plt.bar(x_conv, stage_data, bar_width, bottom=bottom_even, label=stage, color=STAGE_COLORS[stage], edgecolor='black', alpha=0.8)
    bottom_even += np.array(stage_data)

# 绘制FL层
x_fl = np.arange(len(conv_labels), len(conv_labels) + len(fl_labels))
plt.bar(x_fl, fl_values, bar_width, label='namenode fullyconnect', color=STAGE_COLORS['namenode fullyconnect'], edgecolor='black', alpha=0.8)

plt.ylim(0, 0.22)
plt.yticks(np.arange(0, 0.22, 0.03))

all_labels = conv_labels + fl_labels
plt.xticks(np.concatenate((x_conv, x_fl)), all_labels, rotation=45, fontsize=10)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xlabel('Layers', fontsize=12)
plt.title('FL and Conv Layer Timing Distribution (Average of Round 4 & 5)', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# 饼图
# ==========================================
total_time_components = {}

for stage in stage_names_odd:
    total_time_components[stage] = sum(conv_stages_odd[stage])

for stage in stage_names_even:
    total_time_components[stage] = sum(conv_stages_even[stage])

total_time_components['namenode fullyconnect'] = sum(fl_values)

labels_pie = [k for k, v in total_time_components.items() if v > 0]
values_pie = [v for v in total_time_components.values() if v > 0]

plt.figure(figsize=(12, 9))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))
plt.pie(values_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 12})
plt.title('Time Percentage Distribution of FL and Conv Stages', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()
