import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.signal import spectrogram, hamming
from PIL import Image
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from module.Tradition_module import ResNet18, ResNet50, VGG, CNN, L_CNN, LR_CNN, Dual_LRCNN
from module.signal_DAE import LSTM_CNN_SAM

# 设置字体路径（根据你系统中文和英文字体的路径来）
times_path = "/home/ll/.fonts/TimesNewRoman/Times New Roman.ttf"  # 或 Windows: "C:/Windows/Fonts/times.ttf"
simu_path = "/home/ll/.fonts/simsun/simsun.ttc"
times_font = fm.FontProperties(fname=times_path)    # 英文 Times New Roman
simsun_font = fm.FontProperties(fname=simu_path)    # 中文 宋体

# modulation_types = ['OOK', '4ASK', 'BPSK', 'QPSK', '8PSK', '16QAM', 'AM-SSB-SC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
modulation_types = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']


class DualModalityDataset(Dataset):
    def __init__(self, npy_dir, is_train=True, transform=None):
        self.npy_dir = npy_dir
        self.is_train = is_train
        self.transform = transform
        self.data = []

        if is_train:
            for mod in modulation_types:
                npy_mod_path = os.path.join(npy_dir, mod)
                if os.path.isdir(npy_mod_path):
                    for npy_file in os.listdir(npy_mod_path):
                        if npy_file.endswith('.npy'):
                            npy_filepath = os.path.join(npy_mod_path, npy_file)
                            label = mod
                            self.data.append((npy_filepath, label))
        else:
            for mod in modulation_types:
                npy_mod_path = os.path.join(npy_dir, mod)
                if os.path.isdir(npy_mod_path):
                    for snr_folder in os.listdir(npy_mod_path):
                        snr_path = os.path.join(npy_mod_path, snr_folder)
                        if os.path.isdir(snr_path):
                            for npy_file in os.listdir(snr_path):
                                if npy_file.endswith('.npy'):
                                    npy_filepath = os.path.join(snr_path, npy_file)
                                    label = mod
                                    self.data.append((npy_filepath, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_filepath, label = self.data[idx]
        signal = np.load(npy_filepath)
        signal = torch.tensor(signal.T, dtype=torch.float32)

        spec = self.generate_spectrogram(signal)

        label_index = torch.tensor(self.get_label_index(label), dtype=torch.long)

        return signal, spec, label_index, npy_filepath

    def get_label_index(self, label):
        return modulation_types.index(label)

    def generate_spectrogram(self, signal):
        I_signal = signal[0, :]
        Q_signal = signal[1, :]

        window_size = 32
        overlap = window_size // 2

        fs = 256
        h = hamming(window_size)
        f, t, Sxx = spectrogram(I_signal + 1j * Q_signal, fs=fs, window=h, nperseg=window_size, noverlap=overlap)
        Sxx = np.fft.fftshift(Sxx, axes=0)

        Sxx = 10 * np.log10(Sxx + 1e-8)

        Sxx_min, Sxx_max = Sxx.min(), Sxx.max()
        Sxx_normalized = (Sxx - Sxx_min) / (Sxx_max - Sxx_min)
        Sxx_normalized = (Sxx_normalized * 255).astype(np.uint8)

        Sxx_3ch = np.stack([Sxx_normalized] * 3, axis=-1)

        if Sxx_3ch.shape[0] != 224 or Sxx_3ch.shape[1] != 224:
            data = np.array(Image.fromarray(Sxx_3ch).resize((224, 224)))
        else:
            data = Sxx_3ch

        return torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)

# 初始化测试集
test_dir = '../dataset/test/signal'
test_dataset = DualModalityDataset(npy_dir=test_dir, is_train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载模型并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTM_CNN_SAM().to(device)

# 加载模型权重
model.load_state_dict(torch.load('../model.pth', map_location=device))
model.eval()

# 定义保存预测值和真实标签的列表
all_preds = []
all_labels = []

# 推理阶段
with torch.no_grad():
    for signals, specs, labels, _ in test_loader:
        signals, specs, labels = signals.to(device), specs.to(device), labels.to(device)

        # 通过模型获取输出
        # outputs = model(signals, specs)
        outputs, aux = model(signals)

        # 获取预测结果
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        # 保存预测值和真实标签
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 生成混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 可视化并保存不带数字的混淆矩阵
plt.figure(figsize=(10, 8))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modulation_types, yticklabels=modulation_types, cbar=True, annot_kws={"size": 12, "fontproperties": times_font})

# 设置横纵坐标刻度字体大小
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=times_font, fontsize=16)  # 设置横坐标字体大小
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=times_font, fontsize=16)  # 设置纵坐标字体大小

# 设置横纵坐标标签和标题，并指定字体大小
plt.xlabel("预测标签", fontsize=20, fontproperties=simsun_font)
plt.ylabel("真实标签", fontsize=20, fontproperties=simsun_font)
plt.title("分类任务混淆矩阵", fontsize=16)

# 保存图像
plt.savefig('hunxiao.png', dpi=300, bbox_inches='tight')
plt.show()
