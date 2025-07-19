import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"

times_font = fm.FontProperties(fname=times_path)    # 英文 Times New Roman

# 读取分类结果
snr_values = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
accuracy = {
    'Backbone+HAM': [4.37, 4.07, 4.87, 4.87, 7.08, 11.52, 16.55, 22.85, 31.93, 42.40, 53.03, 63.42, 73.58, 81.67, 84.80, 87.02],
    'Backbone+4LSTM': [4.48, 4.50, 5.35, 5.85, 7.77, 13.02, 17.70, 24.80, 34.20, 41.97, 54.33, 63.08, 74.13, 83.73, 87.68, 88.43],
    'Backbone+4LSTM+CABM': [4.42, 4.52, 4.40, 5.15, 6.82, 9.75, 17.13, 24.95, 32.23, 40.67, 53.53, 62.10, 73.62, 82.63, 88.62, 88.77],
    'Backbone+2LSTM+HAM': [4.57, 4.32, 4.95, 5.63, 8.42, 11.93, 18.23, 23.78, 34.38, 43.55, 54.53, 63.63, 74.53, 83.30, 87.65, 89.12],
    'Backbone+4LSTM+HAM': [4.55, 4.25, 5.08, 5.22, 8.52, 12.22, 18.35, 24.05, 33.77, 42.67, 55.03, 64.37, 75.40, 83.90, 88.08, 89.73]
}

# 绘制图像
fig, ax = plt.subplots(figsize=(7, 4))

# 设置颜色
colors = {
    'Backbone+HAM': 'orange',
    'Backbone+4LSTM': 'blue',
    'Backbone+4LSTM+CABM': 'green',
    'Backbone+2LSTM+HAM': 'purple',
    'Backbone+4LSTM+HAM': 'red'
}

# 绘制每个模型的折线并标注数据点
for model in accuracy:
    ax.plot(snr_values, accuracy[model], label=model, color=colors[model], marker='o')

# 设置图例
ax.legend(loc='lower right', fontsize=16, prop=times_font)

# 设置坐标轴标签
ax.set_xlabel('SNR (dB)', fontsize=16, fontproperties=times_font)
ax.set_ylabel('Acc (%)', fontsize=16, fontproperties=times_font)

# 设置坐标轴刻度字体
ax.set_xticks(snr_values)
ax.set_xticklabels([str(snr) for snr in snr_values], fontproperties=times_font, fontsize=16)
ax.set_yticklabels([str(int(tick)) for tick in ax.get_yticks()], fontproperties=times_font, fontsize=16)
ax.text(0.5, -0.2, "(b)", fontproperties=times_font, fontsize=16, ha='center', va='top', transform=ax.transAxes)

# 不显示网格
ax.grid(False)

# 放大右上角的最后几个点
n = -5
axins = inset_axes(ax, width="40%", height="60%", loc='upper left')
for model in accuracy:
    axins.plot(snr_values[n:], accuracy[model][n:], label=model, color=colors[model], marker='o')

# 设置小图刻度
axins.set_xticks(snr_values[n:])
axins.set_xticklabels([str(i) for i in snr_values[n:]], fontproperties=times_font, fontsize=16)
axins.set_yticks([round(i, 1) for i in accuracy['Backbone+HAM'][n:]])
axins.set_yticklabels([round(i, 1) for i in accuracy['Backbone+HAM'][n:]], fontproperties=times_font, fontsize=16)

# 小图刻度朝内
axins.tick_params(axis='both', which='both', labelsize=16, direction='in', length=6)
axins.xaxis.set_ticks_position('bottom')
axins.yaxis.set_ticks_position('right')
axins.grid(True)

# 布局与保存
plt.tight_layout()
plt.savefig('24_ablation.png', dpi=900)

plt.show()