import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import random
import matplotlib.pyplot as plt


# 创建自定义Dataset类
class IQSignalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 定义卷积自编码器
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 2, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# 训练函数
def train_denoising_network(model, train_loader, num_epochs=100, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, noisy_signal in enumerate(train_loader):
            noisy_signal = noisy_signal.float().to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(noisy_signal)
            loss = criterion(outputs, noisy_signal)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.6f}')


# 从文件夹中加载信号
def load_signals_from_folders(dataset_path, modulation_types):
    signals = []
    for modulation in modulation_types:
        modulation_dir = os.path.join(dataset_path, modulation)
        npy_files = [f for f in os.listdir(modulation_dir) if f.endswith('.npy')]

        for npy_file in npy_files:
            # 加载I/Q信号
            iq_data = np.load(os.path.join(modulation_dir, npy_file))
            iq_data = iq_data.T  # Shape: (1024, 2)
            signals.append(iq_data)
    return np.array(signals)

# 使用模型进行降噪
def denoise_signal(model, noisy_signal):
    model.eval()
    with torch.no_grad():
        noisy_signal = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).to(device)
        denoised_signal = model(noisy_signal)
        return denoised_signal.squeeze(0).cpu().numpy()

# 绘制输入信号与输出信号
def plot_signals(noisy_signal, denoised_signal):
    fig, axes = plt.subplots(2, 1, figsize=(15, 6))

    # 绘制I/Q信号
    axes[0].plot(noisy_signal[0], label='Noisy I')
    axes[0].plot(denoised_signal[0], label='Denoised I')
    axes[0].set_title('I Component')
    axes[0].legend()

    axes[1].plot(noisy_signal[1], label='Noisy Q')
    axes[1].plot(denoised_signal[1], label='Denoised Q')
    axes[1].set_title('Q Component')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# 数据准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_path = '../dataset/train/signal'  # 修改为你的数据集路径
modulation_types = [
    'OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
    '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM',
    '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC',
    'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK'
]

signals = load_signals_from_folders(dataset_path, modulation_types)
signals = torch.tensor(signals, dtype=torch.float32)
dataset = IQSignalDataset(signals)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型训练
model = DenoisingAutoencoder().to(device)
train_denoising_network(model, train_loader, num_epochs=10, learning_rate=1e-3)

# 示例：选择一个样本并进行降噪
sample_idx = 0
noisy_sample = signals[sample_idx].numpy()
denoised_sample = denoise_signal(model, noisy_sample)

# 绘制输入信号与输出信号
plot_signals(noisy_sample, denoised_sample)
