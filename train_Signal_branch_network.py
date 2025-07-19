import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm  # 用于显示进度条
import os
from torch.utils.data import Dataset
import numpy as np

from module.DAE_dual_Branch import DAE_dual_branch
from module.CNN_LSTM_MUTI import DAE_dual_branch
from module.Tradition_module import ResNet18, ResNet50, VGG, CNN, L_CNN, LR_CNN, Dual_LRCNN
from module.signal_DAE import DAE_Res_fusion, DAE_Res_fusion2, DAE_Res_fusion_resnet, LSTM_CNN_SAM
from module.signal_DAE_conv2d import DAE_Res_fusion3

modulation_types = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
log_file = 'training_log.txt'

def extract_snr_modulation_from_path(file_path):
    """
    从文件路径中提取SNR和调制方式，假设路径格式为 'test/modulation_type/SNR/data.npy'
    """
    parts = file_path.split(os.sep)
    modulation = parts[-3]  # 倒数第三个部分是调制方式
    snr = parts[-2]         # 倒数第二个部分是SNR文件夹
    return snr, modulation

class DualModalityDataset(Dataset):
    def __init__(self, npy_dir, is_train=True, transform=None):
        """
        npy_dir: 信号npy文件存放路径
        is_train: 如果为True，加载train数据，不区分SNR；否则加载test数据，区分SNR
        transform: 图像预处理操作
        """
        self.npy_dir = npy_dir
        self.is_train = is_train  # 是否为训练集
        self.transform = transform
        self.data = []

        # 处理train数据，并引入SNR数据（建议：可以按一定比例随机采样不同SNR）
        if is_train:
            for mod in modulation_types:
                npy_mod_path = os.path.join(npy_dir, mod)
                if os.path.isdir(npy_mod_path):
                    for npy_file in os.listdir(npy_mod_path):
                        if npy_file.endswith('.npy'):
                            npy_filepath = os.path.join(npy_mod_path, npy_file)
                            label = mod  # 用调制类型作为label
                            self.data.append((npy_filepath, label))

        # 处理test数据（区分SNR）
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
                                    label = mod  # 用调制类型作为label
                                    self.data.append((npy_filepath, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        npy_filepath, label = self.data[idx]
        signal = np.load(npy_filepath)  # 读取npy信号 (1024, 2)，包括I和Q信号
        signal = torch.tensor(signal.T, dtype=torch.float32)  # 转置为 (2, 1024)

        # 将标签转换为索引
        label_index = torch.tensor(self.get_label_index(label), dtype=torch.long)

        return signal, label_index, npy_filepath

    def get_label_index(self, label):
        # 将调制方式转换为索引
        return modulation_types.index(label)

def train(model, device, train_loader, optimizer, classifier_criterion, denoise_criterion, epoch, scheduler = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Training]')  # 进度条
    for signals, labels, npy_path in progress_bar:
        signals = signals.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        output, aux = model(signals) #自己的模型
        # output = model(signals) #自己的模型
        loss = classifier_criterion(output, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条显示
        progress_bar.set_postfix({
            'loss': f'{running_loss / (len(train_loader)):.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    # Update learning rate every 10 epochs
    if scheduler is not None and (epoch + 1) % 10 == 0:
            scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    epoch_time = time.time() - start_time

    # 将结果保存到日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} [Training] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s\n')

    return epoch_acc  # 返回训练准确率

def validate(model, device, test_loader, criterion, epoch, model_path = None):
    # 如果提供了模型路径，则加载模型参数
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    mod_snr_correct = {}  # 用于存储不同调制方式下不同SNR的正确预测数
    mod_snr_total = {}    # 用于存储不同调制方式下不同SNR的总数
    start_time = time.time()

    # 定义 SNR 范围，按顺序排列
    snr_range = list(range(-20, 12, 2))

    # 确保每种调制方式和每种SNR的统计字典初始化
    for mod in modulation_types:
        mod_snr_correct[mod] = {snr: 0 for snr in snr_range}
        mod_snr_total[mod] = {snr: 0 for snr in snr_range}

    progress_bar = tqdm(test_loader, desc=f'Epoch {epoch} [Validation]')  # 进度条
    with torch.no_grad():
        for signals, labels, file_paths in progress_bar:  # 假设loader中包含文件路径信息
            signals = signals.to(device)
            labels = labels.to(device)

            outputs, aux = model(signals) #自己的模型
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 从路径中提取 SNR 和调制方式
            for i, file_path in enumerate(file_paths):
                try:
                    snr, modulation = extract_snr_modulation_from_path(file_path)

                    # 确保 snr 是一个有效的整数值
                    snr_value = int(snr)

                    if modulation in mod_snr_total:
                        mod_snr_total[modulation][snr_value] += 1
                        if predicted[i] == labels[i]:
                            mod_snr_correct[modulation][snr_value] += 1
                except ValueError:
                    print(f"Warning: Invalid SNR value '{snr}' extracted from file path '{file_path}'")

            # 更新进度条显示
            progress_bar.set_postfix({
                'val_loss': f'{test_loss / len(test_loader):.4f}',
                'val_acc': f'{100 * correct / total:.2f}%'
            })

    val_loss = test_loss / len(test_loader)
    val_acc = 100 * correct / total
    val_time = time.time() - start_time

    # 计算每种调制方式和 SNR 的准确率
    mod_snr_accs = {}
    for modulation in mod_snr_correct:
        mod_snr_accs[modulation] = []
        for snr in snr_range:
            if mod_snr_total[modulation][snr] > 0:
                acc = 100 * mod_snr_correct[modulation][snr] / mod_snr_total[modulation][snr]
                mod_snr_accs[modulation].append(f'{acc:.2f}%')
            else:
                mod_snr_accs[modulation].append('--')  # 用占位符表示无数据

    # 计算平均准确率
    total_acc_sum = 0
    total_acc_count = 0
    for modulation in mod_snr_correct:
        for snr in snr_range:
            if mod_snr_total[modulation][snr] > 0:
                total_acc_sum += 100 * mod_snr_correct[modulation][snr] / mod_snr_total[modulation][snr]
                total_acc_count += 1

    avg_acc = total_acc_sum / total_acc_count if total_acc_count else 0

    # 计算每个 SNR 下的所有信号种类的准确率
    snr_accuracy = {}
    for snr in snr_range:
        snr_correct = 0
        snr_total = 0
        for mod in modulation_types:
            snr_correct += mod_snr_correct[mod][snr]
            snr_total += mod_snr_total[mod][snr]
        if snr_total > 0:
            snr_accuracy[snr] = 100 * snr_correct / snr_total
        else:
            snr_accuracy[snr] = None

    # 打印每种调制方式和 SNR 的准确率
    print(f'Modulation-wise and SNR-wise Accuracy (SNRs from {snr_range[0]} to {snr_range[-1]}):')
    for modulation in mod_snr_accs:
        snr_acc_line = '\t'.join(mod_snr_accs[modulation])
        print(f'{modulation}:\t{snr_acc_line}')

    print(f'Average Accuracy: {avg_acc:.2f}%')

    # 打印每个 SNR 下所有信号种类的准确率
    print(f'Overall Accuracy for each SNR:')
    overall_snr_acc_line = []
    for snr in snr_range:
        if snr_accuracy[snr] is not None:
            overall_snr_acc_line.append(f'{snr}:{snr_accuracy[snr]:.2f}%')
        else:
            overall_snr_acc_line.append(f'SNR {snr}: --')
    overall_snr_acc_line_str = '\t'.join(overall_snr_acc_line)
    print(overall_snr_acc_line_str)

    # 将验证结果保存到日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch} [Validation] - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Time: {val_time:.2f}s\n')
        f.write(f'Modulation-wise and SNR-wise Accuracy (SNRs from {snr_range[0]} to {snr_range[-1]}):\n')
        for modulation in mod_snr_accs:
            snr_acc_line = '\t'.join(mod_snr_accs[modulation])
            f.write(f'{modulation}:\t{snr_acc_line}\n')
        f.write(f'Average Accuracy: {avg_acc:.2f}%\n')

        # 保存每个 SNR 下所有信号种类的准确率
        f.write(f'\nOverall Accuracy for each SNR:\n')
        f.write(overall_snr_acc_line_str + '\n')

    return val_acc, avg_acc  # 返回验证准确率和平均准确率

def test_train_loader_accuracy(model, device, train_loader, model_path='best_model_grame.pth'):
    # 如果提供了模型路径，则加载模型参数
    if model_path:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model parameters from {model_path}")


    model.eval()
    correct = 0
    total = 0
    mod_correct = {}  # 用于存储每种调制方式的正确预测数
    mod_total = {}    # 用于存储每种调制方式的总数

    # 初始化每种调制方式的统计字典
    for mod in modulation_types:
        mod_correct[mod] = 0
        mod_total[mod] = 0

    with torch.no_grad():
        for signals, images, labels, _ in train_loader:
            signals, images = signals.to(device), images.to(device)
            labels = labels.to(device)

            # 预测输出
            outputs = model(signals, images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 统计每种调制方式的正确预测数
            for i, label in enumerate(labels):
                label_str = modulation_types[label.item()]  # 将数字标签转换为调制类型字符串
                mod_total[label_str] += 1
                if predicted[i] == label:
                    mod_correct[label_str] += 1

    # 计算每种调制方式的准确率
    mod_accs = {}
    for mod in modulation_types:
        if mod_total[mod] > 0:
            mod_accs[mod] = 100 * mod_correct[mod] / mod_total[mod]
        else:
            mod_accs[mod] = 0

    # 计算平均准确率
    avg_acc = 100 * correct / total if total > 0 else 0

    # 打印每种调制方式的准确率
    print('Modulation-wise Accuracy:')
    for mod, acc in mod_accs.items():
        print(f'{mod}: {acc:.2f}%')

    print(f'Average Accuracy: {avg_acc:.2f}%')

    return mod_accs, avg_acc  # 返回每种调制方式的准确率和平均准确率

def save_best_model(model, best_acc, current_acc, epoch, save_path = 'best_model_grame.pth'):
    """保存表现最好的模型"""
    if current_acc > best_acc:
        torch.save(model.state_dict(), save_path)
        print(f"Best model updated at epoch {epoch} with accuracy {current_acc:.2f}%")
        return current_acc
    return best_acc

def main():
    # 训练和测试数据集路径
    train_dir = 'dataset/train/signal'
    test_dir = 'dataset/test/signal'

    # 创建数据集
    train_dataset = DualModalityDataset(npy_dir=train_dir, is_train=True)
    test_dataset = DualModalityDataset(npy_dir=test_dir, is_train=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM_CNN_SAM(num_classes=24, lstm_layers=2).to(device) #24分类任务

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    classifier_criterion = nn.CrossEntropyLoss()
    denoise_criterion = nn.MSELoss()

    best_acc = 0  # 记录最优验证准确率

    # 清空日志文件
    with open(log_file, 'w') as f:
        f.write("Training Log\n")

    for epoch in range(1, 101):
        train_acc = train(model, device, train_loader, optimizer, classifier_criterion, denoise_criterion, epoch, scheduler)

        # 每 10 轮验证一次
        if epoch == 1 or (epoch % 5 == 0 and epoch < 30) or epoch >= 1:
            val_acc, avg_snr_acc = validate(model, device, test_loader, classifier_criterion, epoch)
            best_acc = save_best_model(model, best_acc, avg_snr_acc, epoch)

if __name__ == '__main__':
    main()
