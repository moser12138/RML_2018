import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib import rcParams

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"

times_font = fm.FontProperties(fname=times_path)    # 英文 Times New Roman

# 读取分类结果
snr_values = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
accuracy = {
    'THMNet': [4.55, 4.25, 5.08, 5.22, 8.52, 12.22, 18.35, 24.05, 33.77, 42.67, 55.03, 64.37, 75.40, 83.90, 88.08, 89.73],
    'L_CNN': [4.13, 4.52, 4.62, 5.47, 7.20, 10.27, 14.88, 21.68, 31.00, 41.23, 51.12, 57.45, 62.42, 65.45, 67.03, 66.83],
    'LR_CNN': [4.22, 4.68, 4.38, 4.80, 6.35, 9.77, 15.98, 23.55, 32.38, 43.03, 53.68, 62.55, 73.62, 82.45, 85.95, 87.15],
    'ResNet': [4.15, 4.37, 4.37, 4.88, 8.18, 11.82, 16.73, 22.88, 32.37, 41.68, 53.98, 62.95, 73.93, 82.27, 86.38, 88.12],
    'VGG': [4.32, 4.07, 4.25, 5.38, 7.52, 12.87, 19.05, 25.28, 33.27, 42.63, 52.90, 60.02, 71.68, 77.48, 79.97, 79.77],
}

# 绘制图像
plt.figure(figsize=(6, 4))

# 设置不同颜色，THMNet 为红色
colors = {
    'THMNet': 'red',
    'L_CNN': 'blue',
    'LR_CNN': 'green',
    'ResNet': 'purple',
    'VGG': 'orange'
}

# 绘制每个模型的折线并标注数据点
for model in accuracy:
    plt.plot(snr_values, accuracy[model], label=model, color=colors[model], marker='o')  # marker='o' 表示每个数据点

# 设置图例
plt.legend(loc='lower right', fontsize=30, prop=times_font)  # 图例位置和字体大小

# # 设置坐标轴标签（中文）
plt.xlabel('SNR (dB)',fontproperties=times_font, fontsize=16)
plt.ylabel('Acc (%)',fontproperties=times_font, fontsize=16)

# 设置标题
# plt.title('模型分类准确率对比', fontsize=16)

# 调整坐标轴刻度数字大小
plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标数字的字体大小为10
plt.xticks(fontproperties=times_font, fontsize=16)  # x轴刻度
plt.yticks(fontproperties=times_font, fontsize=16)  # y轴刻度

# 显示网格
plt.grid(True)

# 保存图像到文件
plt.tight_layout()
plt.savefig('24_compare.png', dpi=1200)

# 显示图像
plt.show()
