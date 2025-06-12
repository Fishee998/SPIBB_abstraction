import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

# 读取完整数据
df = pd.read_excel('/Users/yuan/codes/thesis-experiments/abstract_SPI/SPIBB/results/Gridworld/results_44_1000.xlsx')

# 全部轨迹数
x_full = df['nb_trajectories']
y_columns = [
    'baseline_perf', 'pi_star_perf', 'perfrl',
    'perf_Pi_b_SPIBB', 'abstract_perf_Pi_b_SPIBB'
    #'perf_Pi_leq_b_SPIBB', 'abstract_perf_Pi_leq_b_SPIBB'
]

label_map = {
    'baseline_perf': r'$\pi_b$',
    'pi_star_perf': r'$\pi^*$',
    'perfrl': r'$\pi_{\mathrm{RL}}$',
    'perf_Pi_b_SPIBB': r'$\pi_b$-SPIBB',
    'abstract_perf_Pi_b_SPIBB': r'Abstract $\Pi_b$-SPIBB',
    #'perf_Pi_leq_b_SPIBB': r'$\pi{\leq}b$-SPIBB',
    #'abstract_perf_Pi_leq_b_SPIBB': r'Abstract $\pi{\leq}b$-SPIBB',
}


markers = ['d', 'd', 'o', 'D', 's']

# 先把所有数据按原序号画出来
fig, ax = plt.subplots(figsize=(6,4))
positions = list(range(len(df)))  # 0,1,2,…,len(df)-1


for col, marker in zip(y_columns, markers):
    if col == 'baseline_perf':
        ax.plot(positions, df[col], linestyle='--', label=label_map[col])
    elif col == 'pi_star_perf':
        ax.plot(positions, df[col], linestyle='-.', label=label_map[col])
    else:
        ax.plot(positions, df[col], marker=marker, linestyle='-', label=label_map[col])



"""
for col, marker in zip(y_columns, markers):
    ax.plot(positions, df[col], marker=marker, linestyle='-', label=col)
"""
#for col in y_columns:
#    ax.plot(positions, df[col], marker='o', label=col)

# 然后只在这七个位置打上刻度标签
ticks = [5, 10, 50, 200, 1000, 5000, 10000]
# 找到它们在整个 df 中对应的索引
tick_positions = [int(df.index[df['nb_trajectories']==t][0]) for t in ticks]

ax.set_xticks(tick_positions)
ax.set_xticklabels(ticks)

ax.set_xlabel('Number of Trajectories')
ax.set_ylabel('Performance')
ax.set_ylim(3.5,5.5)
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(loc='best')
plt.tight_layout()
plt.show()
