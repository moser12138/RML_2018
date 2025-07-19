import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from module.signal_DAE import LSTM_CNN_SAM
from matplotlib import font_manager as fm

# 设置 Times New Roman 字体路径
times_path = "/home/.fonts/TimesNewRoman/Times New Roman.ttf"  # 根据你系统的字体路径调整
times_font = fm.FontProperties(fname=times_path)

# **调制类型定义**
modulation_types = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
label_map = {mod: i for i, mod in enumerate(modulation_types)}

# **模型加载**
model_path = "../result/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM_CNN_SAM()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# **数据路径 & 目标 SNR**
data_path = "../dataset/test/signal"
snr_values = [-20, -16, -12, -8, -4, 0, 4, 8, 10]  # 需要可视化的 SNR

# **颜色映射**
colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(modulation_types)))

# **设置子图布局**
num_snr = len(snr_values)
cols = 3  # 每行显示 3 张图
rows = (num_snr + cols - 1) // cols  # 计算行数

fig, axes = plt.subplots(rows, cols, figsize=(2.3 *cols, 2 * rows))
axes = axes.flatten()  # 展平以便索引

# **t-SNE 降维 & 可视化**
for i, snr in enumerate(snr_values):
    snr_str = str(snr)
    data_list, labels = [], []

    for mod in modulation_types:
        mod_path = os.path.join(data_path, mod)
        snr_path = os.path.join(mod_path, snr_str)

        if not os.path.exists(snr_path):
            continue

        for file in os.listdir(snr_path):
            if file.endswith(".npy"):
                file_path = os.path.join(snr_path, file)
                signal_data = np.load(file_path)
                if signal_data.shape[0] == 2 * 1024:
                    signal_data = signal_data.reshape(-1)

                data_list.append(signal_data)
                labels.append(label_map[mod])

    if len(data_list) == 0:
        print(f"Warning: No data found for SNR = {snr}")
        continue

    # **转换数据格式**
    data_array = np.array(data_list, dtype=np.float32)
    labels_array = np.array(labels)

    # **特征提取**
    with torch.no_grad():
        batch_size = 128
        features_list = []

        inputs_tensor = torch.tensor(data_array).view(-1, 2, 1024)  # 先在 CPU reshape

        for start_idx in range(0, len(inputs_tensor), batch_size):
            end_idx = start_idx + batch_size
            batch = inputs_tensor[start_idx:end_idx].to(device)

            with torch.no_grad():
                batch_features, _ = model(batch)

            features_list.append(batch_features.cpu())

        features = torch.cat(features_list, dim=0).numpy()

    # **t-SNE 降维**
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)

    # **绘制 t-SNE**
    ax = axes[i]
    for mod_idx, mod in enumerate(modulation_types):
        indices = (labels_array == mod_idx)
        if np.sum(indices) > 0:
            ax.scatter(features_2d[indices, 0], features_2d[indices, 1],
                       color=colors[mod_idx], label=mod, alpha=0.6, s=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"SNR = {snr}", fontproperties=times_font, fontsize=20)

# **添加统一图例，确保图例使用 Times New Roman 字体并设置字体大小**
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=6)
           for i in range(len(modulation_types))]

# **设置图例字体和字体大小，并确保图例使用 Times New Roman 字体**
fig.legend(handles, modulation_types, loc="lower center", prop=times_font, fontsize=20, ncol=6, frameon=True)

# **调整布局 & 保存**
plt.tight_layout(rect=[0, 0.07, 1, 1])  # 留出图例位置
plt.savefig("11_tsne.png", dpi=300)  # **保存图片**
plt.show()
