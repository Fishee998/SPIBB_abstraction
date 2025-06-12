import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 Excel 文件
#df = pd.read_excel("/Users/yuan/codes/thesis-experiments/abstract_SPI/SPIBB/results/Gridworld/results_7.xlsx")


numbers = [5, 7, 10, 15, 20, 30, 50, 70, 100]

# 用于存储每个文件读取的DataFrame
dfs = []

for num in numbers:
    filepath = f"/Users/yuan/codes/thesis-experiments/abstract_SPI/SPIBB/results/Gridworld/results_{num}.xlsx"
    df = pd.read_excel(filepath)
    df['file_index'] = num  # 添加一列标记来源
    dfs.append(df)

# 合并所有DataFrame
all_data = pd.concat(dfs, ignore_index=True)
all_data.to_csv("all_data.csv", index=False)



# 要展示的策略性能列
perf_columns = [
    'baseline_perf',
    'pi_star_perf',
    'perfrl',
    'perf_Pi_b_SPIBB',
    'abstract_perf_Pi_b_SPIBB'
]

# 方法名映射（用于图例更清晰）
method_label_map = {
    'baseline_perf': r'$\pi_b$',
    'pi_star_perf': r'$\pi^*$',
    'perfrl': r'${\mathrm{Basic RL}}$',
    'perf_Pi_b_SPIBB': r'$\pi_b$-SPIBB',
    'abstract_perf_Pi_b_SPIBB': r'Abstract $\pi_b$-SPIBB',
}

custom_colors = {
    'baseline_perf': '#f27830',       # 蓝色
    'pi_star_perf': '#98a1b1',        # 橙色
    'perfrl': '#9279b3',              # 绿色
    'perf_Pi_b_SPIBB': '#b3ce8f',     # 红色
    'abstract_perf_Pi_b_SPIBB': '#6db5c4'  # 紫色
}

# 每条线的样式（线型 + 标记）
line_styles = ['--' ,(0, (3, 1, 1, 1)), ':', ':', ':']
markers = ['o', 's', 'o', 's', 'D']  # 圆形、方形、菱形、三角上、三角下
palette = sns.color_palette("tab10", n_colors=len(perf_columns))

for df in dfs:

    # 转换为长格式
    df_long = df.melt(
        id_vars=['nb_trajectories', 'seed'],
        value_vars=perf_columns,
        var_name='method',
        value_name='performance'
    )
    # 创建画布
    plt.figure(figsize=(5, 4))

    # 按方法逐条绘制
    for i, method in enumerate(perf_columns):
        method_data = df_long[df_long['method'] == method]
        if method == 'baseline_perf':
            sns.lineplot(
                data=method_data,
                x='nb_trajectories',
                y='performance',
                label=method_label_map[method],
                estimator='mean',
                ci='sd',
                linewidth=2,
                linestyle=line_styles[i],
                #color=palette[i]
                color = custom_colors[method]
            )
        elif method == 'pi_star_perf':
            sns.lineplot(
                data=method_data,
                x='nb_trajectories',
                y='performance',
                label=method_label_map[method],
                estimator='mean',
                ci='sd',
                linewidth=2,
                linestyle=line_styles[i],
                #color=palette[i]
                color = custom_colors[method]
            )
        else:
            sns.lineplot(
                data=method_data,
                x='nb_trajectories',
                y='performance',
                label=method_label_map[method],
                estimator='mean',
                ci='sd',
                linewidth=1,
                marker=markers[i],
                linestyle=line_styles[i],
                #color=palette[i]
                color = custom_colors[method]
            )

    # 图形美化
    #plt.title('Performance vs Number of Trajectories')
    plt.xlabel('Number of Trajectories')
    plt.ylabel('Mean Performance')
    plt.xscale('log')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.gca().get_legend().remove()
    plt.legend(title='Method')

    N_wedge_value = df['N_wedge'].iloc[0]
    plt.savefig(f"N{N_wedge_value}.svg", format='pdf', bbox_inches='tight')
    #plt.savefig("N7.pdf", format='pdf', bbox_inches='tight')

    plt.show()
