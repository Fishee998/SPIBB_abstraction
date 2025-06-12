import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取原始数据（每一行是一次试验的 return）
df = pd.read_csv("all_data.csv")  # 包含列：nb_trajectories, N_wedge, method, return

# 设定 CVaR 置信水平 α
alpha = 0.01

# 定义计算 CVaR 的函数
def cvar(series, alpha=0.01):
    sorted_vals = series.sort_values()
    cutoff_index = max(1, int(len(sorted_vals) * alpha))  # 至少取1个
    return sorted_vals.iloc[:cutoff_index].mean()

# 只画某一种方法的 CVaR（如 Abstract SPIBB）
target_method = "abstract_perf_Pi_b_SPIBB"

# 筛选目标方法数据
filtered_df = df[df['method'] == target_method]

# 按 (nb_trajectories, N_wedge) 分组计算 CVaR
cvar_df = (
    filtered_df
    .groupby(['N_wedge', 'nb_trajectories'])['return']
    .apply(lambda x: cvar(x, alpha=alpha))
    .reset_index(name='cvar')
)

# 构造热力图数据
pivot = cvar_df.pivot(index='N_wedge', columns='nb_trajectories', values='cvar')

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title(f"{int(alpha*100)}%-CVaR Heatmap: {target_method}")
plt.xlabel("Number of Trajectories")
plt.ylabel("N_wedge")
plt.tight_layout()
plt.savefig(f"{target_method}_CVaR_{int(alpha*100)}.pdf")
plt.show()
