import matplotlib.pyplot as plt
import numpy as np

from time_data import data_round4, data_round5

# 设置中文字体支持 (防止乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 计算平均值
def calc_mean(val1, val2):
    if isinstance(val1, list):
        return [(val1 + val2) / 2 for val1, val2 in zip(val1, val2)]
    else:
        return (val1 + val2) / 2

avg_data = {key: calc_mean(data_round4[key], data_round5[key]) for key in data_round4}

# ==========================================
# 2. 数据重组：构建 FL 和 Conv 结构
# ==========================================

# FL1-3 (全连接层)
fl_labels = ['FL1', 'FL2', 'FL3']
fl_values = avg_data['namenode_fullyconnect']

# Conv1-10 (卷积层)
conv_labels = [f'Conv{i+1}' for i in range(10)]

# 定义阶段名称 (顺序对应从上到下)
stage_names = [
    'namenode pre_send', 
    'datanode pre_conv',    # <--- 已更新为列表数据
    'datanode conv_time', 
    'datanode pre_send', 
    'datanode transfer_time', 
    'namenode after_receive', 
    'namenode maxpool'
]

# 初始化存储结构
conv_stages_data = {stage: [] for stage in stage_names}

# 填充数据
for i in range(10):
    # 1. namenode pre_send
    conv_stages_data['namenode pre_send'].append(avg_data['namenode_pre_send'][i])
    
    # 2. datanode pre_conv (现在是列表，直接取第i项)
    conv_stages_data['datanode pre_conv'].append(avg_data['datanode_pre_conv'][i])
    
    # 3. datanode conv_time
    conv_stages_data['datanode conv_time'].append(avg_data['datanode_conv_time'][i])
    
    # 4. datanode pre_send
    conv_stages_data['datanode pre_send'].append(avg_data['datanode_pre_send'][i])
        
    # 5. datanode transfer_time
    conv_stages_data['datanode transfer_time'].append(avg_data['datanode_transfer_time'][i])
    
    # 6. namenode after_receive
    conv_stages_data['namenode after_receive'].append(avg_data['namenode_after_receive'][i])
    
    # 7. namenode maxpool
    conv_stages_data['namenode maxpool'].append(avg_data['namenode_maxpool'][i])

# ==========================================
# 3. 生成柱状图 (Bar Chart)
# ==========================================
plt.figure(figsize=(20, 8))

# 绘制 Conv 部分 (堆叠柱状图)
bar_width = 0.6
x_conv = np.arange(len(conv_labels))
bottom = np.zeros(len(conv_labels))

# 使用更丰富的颜色映射
colors = plt.cm.tab20(np.linspace(0, 1, len(stage_names)))

for idx, stage in enumerate(stage_names):
    plt.bar(x_conv, conv_stages_data[stage], bar_width, bottom=bottom, label=stage, color=colors[idx], edgecolor='black', alpha=0.8)
    bottom += np.array(conv_stages_data[stage])

# 绘制 FL 部分 (并列柱状图)
x_fl = np.arange(len(conv_labels), len(conv_labels) + len(fl_labels))
plt.bar(x_fl, fl_values, bar_width, label='namenode fullyconnect', color='orange', edgecolor='black', alpha=0.8)

# 设置标签和标题
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
# 4. 生成饼图 (Pie Chart)
# ==========================================
total_time_components = {}

# 汇总 Conv 各阶段总和
for stage in stage_names:
    total_time_components[stage] = sum(conv_stages_data[stage])

# 汇总 FL 阶段总和
total_time_components['namenode fullyconnect'] = sum(fl_values)

# 过滤掉时间为0的项
labels_pie = [k for k, v in total_time_components.items() if v > 0]
values_pie = [v for v in total_time_components.values() if v > 0]

# 绘制饼图
plt.figure(figsize=(12, 9))
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels_pie)))
plt.pie(values_pie, labels=labels_pie, autopct='%1.1f%%', startangle=140, colors=colors_pie, textprops={'fontsize': 12})
plt.title('Time Percentage Distribution of FL and Conv Stages', fontsize=14)
plt.axis('equal') 
plt.tight_layout()
plt.show()