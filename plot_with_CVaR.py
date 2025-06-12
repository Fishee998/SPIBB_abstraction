import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 Seaborn 风格
sns.set(style="whitegrid")

# 读取 CSV 数据（修改为你自己的路径）
data_path = "all_data.csv"
df = pd.read_csv(data_path)

# 选择要分析的方法列
methods = ['perf_Pi_b_SPIBB', 'abstract_perf_Pi_b_SPIBB']

# 按 trajectory 和 N_wedge 分组求平均值
grouped = df.groupby(['nb_trajectories', 'N_wedge'])[methods].mean().reset_index()

# 遍历每种方法并画热力图
for method in methods:
    pivot_table = grouped.pivot(index='N_wedge', columns='nb_trajectories', values=method)

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Mean Performance: {method}")
    plt.xlabel("Number of Trajectories")
    plt.ylabel("N_wedge")
    plt.tight_layout()
    plt.savefig(f"{method}_heatmap.pdf")
    plt.close()

# 可选：画差值热力图
#diff_pivot = grouped.pivot(index='N_wedge', columns='nb_trajectories', values=methods[1]) - \
#             grouped.pivot(index='N_wedge', columns='nb_trajectories', values=methods[0])

plt.figure(figsize=(10, 6))
#sns.heatmap(diff_pivot, annot=True, center=0, fmt=".2f", cmap="RdBu")
plt.title("Performance Difference (Abstract - Original)")
plt.xlabel("Number of Trajectories")
plt.ylabel("N_wedge")
plt.tight_layout()
plt.show()
#plt.savefig("performance_difference_heatmap.pdf")
plt.close()

print("热力图绘制完成，图像已保存为 PDF 文件。")
