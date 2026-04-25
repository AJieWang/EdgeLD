import matplotlib.pyplot as plt
import numpy as np

# ---------------------- 1. 原始数据 ----------------------
# 横轴：三种方法
methods = ['OCBP', 'Pooled', 'PABC']
# 第4轮、第5轮总耗时 (单位：秒)
time_round4 = [0.702, 0.421, 0.344]
time_round5 = [0.741, 0.395, 0.369]
# 计算两轮平均耗时（用于计算加速比）
avg_time = [(t4 + t5) / 2 for t4, t5 in zip(time_round4, time_round5)]
# 加速比：以OCBP为基准（OCBP耗时 / 对应方法耗时）
speedup = [avg_time[0] / t for t in avg_time]

# ---------------------- 2. 绘图基础设置 ----------------------
fig, ax1 = plt.subplots(figsize=(10, 7), dpi=120)
bar_width = 0.35  # 柱状图宽度
x = np.arange(len(methods))  # 横轴位置

# ---------------------- 3. 左轴：总耗时柱状图 ----------------------
# 绘制两轮耗时的分组柱状图（匹配原图斜纹填充风格）
bars_round4 = ax1.bar(
    x - bar_width/2, time_round4, bar_width,
    label='Round 4', color='#00cc96', hatch='//'
)
bars_round5 = ax1.bar(
    x + bar_width/2, time_round5, bar_width,
    label='Round 5', color='#ff9933', hatch='//'
)

# 左轴样式配置
ax1.set_xlabel('Method', fontsize=14, fontweight='bold')
ax1.set_ylabel('Total Time (s)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 0.8)  # 适配数据范围
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)  # 横向网格线
ax1.legend(loc='upper left', fontsize=12)

# 在柱状图上标注具体耗时数值
for bar in bars_round4:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2., height + 0.01,
        f'{height:.3f}s', ha='center', va='bottom', fontsize=10
    )
for bar in bars_round5:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2., height + 0.01,
        f'{height:.3f}s', ha='center', va='bottom', fontsize=10
    )

# ---------------------- 4. 右轴：加速比折线图 ----------------------
ax2 = ax1.twinx()  # 共享横轴，创建双纵轴
# 绘制红色虚线+圆点折线（完全匹配原图风格）
ax2.plot(
    x, speedup, color='#ff6666',
    marker='o', linestyle='--', linewidth=3, markersize=10,
    label='Speedup (×)'
)

# 右轴样式配置
ax2.set_ylabel('Speedup (×)', fontsize=14, fontweight='bold', color='#ff6666')
ax2.set_ylim(0, 2.2)  # 适配加速比范围
ax2.tick_params(axis='y', labelcolor='#ff6666', labelsize=12)

# 在折线上标注加速比数值
for i, s in enumerate(speedup):
    ax2.text(
        i, s + 0.05, f'{s:.2f}×',
        ha='center', va='bottom', fontsize=11,
        color='#ff6666', fontweight='bold'
    )

# ---------------------- 5. 标题与布局 ----------------------
plt.title(
    'Total Time and Speedup Comparison of Different Methods',
    fontsize=15, fontweight='bold', pad=20
)
fig.tight_layout()  # 自动调整布局，避免标签被截断
plt.show()