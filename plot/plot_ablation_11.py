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
    'Backbone+HAM': [9.60, 9.31, 10.11, 11.24, 15.89, 24.69, 38.18, 48.95, 67.35, 84.22, 97.31, 99.71, 99.96, 100.00, 100.00, 99.96],
    'Backbone+4LSTM': [8.91, 10.18, 10.00, 13.13, 18.00, 28.18, 39.16, 51.56, 68.80, 82.33, 95.38, 99.31, 99.93, 99.89, 99.96, 99.93],
    'Backbone+4LSTM+CABM': [8.91, 10.29, 10.51, 12.47, 16.84, 26.95, 40.25, 55.05, 69.78, 83.67, 95.93, 99.45, 99.89, 100.00, 100.00, 99.89],
    'Backbone+2LSTM+HAM': [9.93, 10.51, 10.25, 12.55, 17.27, 25.27, 39.27, 51.75, 70.00, 84.18, 96.40, 99.71, 99.96, 99.96, 100.00, 99.93],
    'Backbone+4LSTM+HAM': [9.85, 10.11, 10.80, 12.87, 18.73, 27.64, 40.40, 51.60, 70.33, 83.82, 95.64, 99.45, 99.85, 100.00, 100.00, 99.93]
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
ax.text(0.5, -0.2, "(a)", fontproperties=times_font, fontsize=16, ha='center', va='top', transform=ax.transAxes)

# 不显示网格
ax.grid(False)

# 放大右上角的最后几个点
n = -5
axins = inset_axes(ax, width="40%", height="55%", loc='upper left')
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

# 保存图像
plt.tight_layout()
plt.savefig('11_ablation.png', dpi=900)
plt.show()
