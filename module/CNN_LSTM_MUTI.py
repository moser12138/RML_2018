import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Branch(nn.Module):
    def __init__(self, input_channels=2, lstm_hidden_size=128, num_lstm_layers=2, output_features=256):
        super(CNN_LSTM_Branch, self).__init__()

        # 1D CNN部分
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),  # 输出: (batch_size, 64, 512)
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出: (batch_size, 64, 256)

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # 输出: (batch_size, 128, 128)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),  # 输出: (batch_size, 128, 64)
        )

        # LSTM部分
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, num_layers=num_lstm_layers,
                            batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(lstm_hidden_size * 2, output_features)  # 双向 LSTM 的输出维度是 hidden_size * 2

    def forward(self, x):
        # 输入x形状: (batch_size, 2, 1024)
        x = self.cnn(x)  # 输出形状: (batch_size, 128, 64)

        # 准备LSTM输入
        x = x.permute(0, 2, 1)  # 将x从 (batch_size, 128, 64) 转为 (batch_size, 64, 128)

        # LSTM
        x, _ = self.lstm(x)  # 输出形状: (batch_size, 64, lstm_hidden_size * 2)

        # 取最后一个时间步的输出
        x = x[:, -1, :]  # 输出形状: (batch_size, lstm_hidden_size * 2)

        # 全连接层
        x = self.fc(x)  # 输出形状: (batch_size, 256)

        return x

class MultiBranchNet(nn.Module):
    def __init__(self, output_features=256):
        super(MultiBranchNet, self).__init__()

        # Branch 1: Local Features (Smaller Kernels)
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: 32x224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x56x56
        )

        # Branch 2: Global Features (Larger Kernels)
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),  # Output: 32x224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32x112x112

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # Output: 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x56x56
        )

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=3, stride=1, padding=1),  # Output: 128x56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128x28x28
        )

        # Output 256 Features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, output_features)  # Output: 256
        )

    def forward(self, x):
        # Branch 1
        x1 = self.branch1(x)

        # Branch 2
        x2 = self.branch2(x)

        # Concatenate along the channel dimension
        x = torch.cat((x1, x2), dim=1)  # Output: (batch_size, 64*2, 56, 56)

        # Fusion and output 256 features
        x = self.fusion(x)
        x = self.fc(x)

        return x

# 网络整体
class DAE_dual_branch(nn.Module):
    def __init__(self, num_classes):
        super(DAE_dual_branch, self).__init__()
        # Branch 1: CNN-LSTM for IQ signal
        self.signal = CNN_LSTM_Branch()

        # Branch 2: Multi-branch CNN for spectrogram images
        self.img = MultiBranchNet()

        # 添加可学习的权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始化信号分支权重
        self.beta = nn.Parameter(torch.tensor(0.5))  # 初始化图像分支权重

        # Feature Fusion Layer
        # Concatenate the 256-dimensional features from both branches to form a 512-dimensional feature vector
        self.fusion_fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Output the final classification (num_classes)
        )

    def forward(self, signal_input, img_input):
        # Process the signal input through the CNN-LSTM branch
        signal_features = self.signal(signal_input)  # Output shape: (batch_size, 256)

        # Process the image input through the multi-branch CNN
        img_features = self.img(img_input)  # Output shape: (batch_size, 256)

        # 融合分支 - 通过alpha和beta进行自学习权重融合
        seq_weighted = self.alpha * signal_features  # 乘以可学习的alpha权重
        img_weighted = self.beta * img_features  # 乘以可学习的beta权重
        fused_features = seq_weighted + img_weighted  # 加和两者 (batch_size, 256)权重融合

        # Concatenate features from both branches
        # fused_features = torch.cat((signal_features, img_features), dim=1)  # Output shape: (batch_size, 512)

        # Apply fusion fully connected layer to classify
        output = self.fusion_fc(fused_features)  # Output shape: (batch_size, num_classes)

        return output

# 测试网络
if __name__ == "__main__":
    model = CNN_LSTM_Branch()
    input_tensor = torch.randn(8, 2, 1024)  # (batch_size, input_channels, sequence_length)
    output = model(input_tensor)
    print(output.shape)  # 应输出: torch.Size([8, 256])
