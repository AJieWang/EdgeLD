import matplotlib.pyplot as plt
import numpy as np

from time_data import data_round4, data_round5

# ===================== 统一配置 - 核心修改 =====================
# 1. 全局样式+中文字体（与第一个代码完全一致）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 2. 统一配色（与第一个代码完全相同的固定色值）
STAGE_COLORS = {
    'namenode pre_send': '#1f77b4',
    'datanode pre_conv': '#ff7f0e',
    'datanode conv_time': '#2ca02c',
    'datanode pre_send': '#d62728',
    'datanode transfer_time': '#9467bd',
    'namenode after_receive': '#8c564b',
    'namenode maxpool': '#e377c2',
    'namenode fullyconnect': '#7f7f7f'  # FL层统一灰色
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
conv_labels = ['Conv1', 'Conv2', 'Conv4', 'Conv5', 'Conv7', 'Conv8', 'Conv10', 'Conv11', 'Conv13', 'Conv14']
stage_names_odd = ['namenode pre_send', 'datanode pre_conv', 'datanode conv_time']
stage_names_even = ['datanode pre_send', 'datanode transfer_time', 'namenode after_receive', 'namenode maxpool']

conv_stages_odd = {stage: [] for stage in stage_names_odd}
conv_stages_even = {stage: [] for stage in stage_names_even}

for i in range(10):
    data_idx = i // 2
    if i % 2 == 0:
        conv_stages_odd['namenode pre_send'].append(avg_data['namenode_pre_send'][data_idx])
        conv_stages_odd['datanode pre_conv'].append(avg_data['datanode_pre_conv'][data_idx])
        conv_stages_odd['datanode conv_time'].append(avg_data['datanode_conv_time'][data_idx])
    else:
        conv_stages_even['datanode pre_send'].append(avg_data['datanode_pre_send'][data_idx])
        conv_stages_even['datanode transfer_time'].append(avg_data['datanode_transfer_time'][data_idx])
        conv_stages_even['namenode after_receive'].append(avg_data['namenode_after_receive'][data_idx])
        conv_stages_even['namenode maxpool'].append(avg_data['namenode_maxpool'][data_idx])

fl_labels = ['FL1', 'FL2', 'FL3']
fl_values = avg_data['namenode_fullyconnect']

# ==========================================
# 生成柱状图 (统一颜色+纵轴)
# ==========================================
plt.figure(figsize=(20, 8))
bar_width = 0.6
x_conv = np.arange(len(conv_labels))

# 绘制奇数层柱状图（使用统一配色）
bottom_odd = np.zeros(len(conv_labels))
for idx, stage in enumerate(stage_names_odd):
    stage_data = [conv_stages_odd[stage][i//2] if i % 2 == 0 else 0 for i in range(10)]
    plt.bar(
        x_conv, stage_data, bar_width,
        bottom=bottom_odd, label=stage,
        color=STAGE_COLORS[stage],  # 统一固定颜色
        edgecolor='black', alpha=0.8
    )
    bottom_odd += np.array(stage_data)

# 绘制偶数层柱状图（使用统一配色）
bottom_even = np.zeros(len(conv_labels))
for idx, stage in enumerate(stage_names_even):
    stage_data = [0 if i % 2 == 0 else conv_stages_even[stage][i//2] for i in range(10)]
    plt.bar(
        x_conv, stage_data, bar_width,
        bottom=bottom_even, label=stage,
        color=STAGE_COLORS[stage],  # 统一固定颜色
        edgecolor='black', alpha=0.8
    )
    bottom_even += np.array(stage_data)

# 绘制FL层（使用统一配色）
x_fl = np.arange(len(conv_labels), len(conv_labels) + len(fl_labels))
plt.bar(
    x_fl, fl_values, bar_width,
    label='namenode fullyconnect', color=STAGE_COLORS['namenode fullyconnect'],
    edgecolor='black', alpha=0.8
)

# ===================== 统一纵轴 =====================
plt.ylim(0, 1.01)
plt.yticks(np.arange(0, 1.01, 0.10))

# plt.ylim(0, 0.30)
# plt.yticks(np.arange(0, 0.26, 0.05))

# 通用标签配置（保持原有）
all_labels = conv_labels + fl_labels
plt.xticks(np.concatenate((x_conv, x_fl)), all_labels, rotation=45, fontsize=10)
plt.ylabel('Time (seconds)', fontsize=12)
plt.xlabel('Layers', fontsize=12)
plt.title('FL and Conv Layer Timing Distribution (Average of Round 4 & 5)', fontsize=14)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# # ==========================================
# # 饼图（保持原有逻辑）
# # ==========================================
# total_time_components = {}
# for stage in stage_names_odd:
#     total_time_components[stage] = sum(conv_stages_odd[stage])
# for stage in stage_names_even:
#     total_time_components[stage] = sum(conv_stages_even[stage])
# total_time_components['namenode fullyconnect'] = sum(fl_values)
#
# labels_pie = [k for k, v in total_time_components.items() if v > 0]
# values_pie = [v for v in total_time_components.values() if v > 0]
#
# plt.figure(figsize=(12, 9))
# colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))
# plt.pie(values_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 12})
# plt.title('Time Percentage Distribution of FL and Conv Stages', fontsize=14)
# plt.axis('equal')
# plt.tight_layout()
# plt.show()