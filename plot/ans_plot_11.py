import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"

times_font = fm.FontProperties(fname=times_path)    # 英文 Times New Roman


# 定义数据
snr_levels = np.arange(-20, 12, 2)  # SNR 从 -20 到 10，步长 2
modulations = {
    "OOK": [5.20, 5.60, 8.80, 10.00, 25.20, 50.00, 71.20, 75.60, 87.20, 94.80, 97.60, 100.00, 100.00, 100.00, 100.00, 99.60],
    "4ASK": [1.20, 0.80, 1.60, 5.20, 15.20, 26.00, 40.40, 58.80, 82.80, 91.20, 95.60, 98.40, 98.80, 100.00, 100.00, 99.60],
    "BPSK": [5.20, 4.80, 6.00, 4.00, 5.60, 12.40, 21.60, 38.80, 91.60, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
    "QPSK": [14.00, 11.20, 11.60, 10.40, 11.20, 6.80, 3.60, 2.40, 5.20, 39.20, 93.60, 100.00, 100.00, 100.00, 100.00, 100.00],
    "8PSK": [4.40, 4.80, 4.00, 6.00, 4.80, 1.60, 3.20, 2.00, 12.40, 41.20, 74.80, 96.80, 99.60, 100.00, 100.00, 100.00],
    "16QAM": [5.20, 4.40, 6.00, 5.60, 5.60, 8.00, 15.20, 36.00, 61.20, 88.40, 98.00, 100.00, 100.00, 100.00, 100.00, 100.00],
    "AM-SSB-SC": [9.60, 8.80, 12.40, 19.20, 30.80, 33.20, 41.60, 71.60, 98.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
    "AM-DSB-SC": [12.40, 11.20, 11.20, 17.20, 25.60, 46.80, 86.00, 98.80, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
    "FM": [20.40, 19.60, 22.40, 28.80, 44.80, 66.40, 94.00, 99.60, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
    "GMSK": [17.60, 24.00, 20.80, 22.80, 22.80, 26.40, 32.80, 40.40, 78.80, 99.60, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00],
    "OQPSK": [13.20, 16.00, 14.00, 12.40, 14.40, 26.40, 34.80, 43.60, 56.40, 67.60, 92.40, 98.80, 100.00, 100.00, 100.00, 100.00],
}

# **绘制折线图**
plt.figure(figsize=(9, 6))
for mod, acc in modulations.items():
    plt.plot(snr_levels, acc, marker='o', linestyle='-', label=mod)

# 设置字体为 Times New Roman
plt.xlabel("SNR (dB)", fontsize=16, fontproperties=times_font)
plt.ylabel("Acc (%)", fontsize=16, fontproperties=times_font)
plt.title("(e)", fontproperties=times_font, fontsize=18, y=-0.2)
plt.legend(loc="lower right", fontsize=16, prop=times_font)  # 仅设置字体大小

# 调整坐标轴刻度数字大小
plt.tick_params(axis='both', labelsize=16)  # 设置横纵坐标数字的字体大小为10
plt.xticks(fontproperties=times_font, fontsize=16)  # x轴刻度
plt.yticks(fontproperties=times_font, fontsize=16)  # y轴刻度
plt.grid(True)

# 保存图像到文件
plt.tight_layout()
plt.savefig('11_ans.png', dpi=900)
# 显示图像
plt.show()