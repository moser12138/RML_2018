import torch
import torch.nn as nn
import torch.nn.functional as F

# from .De_noise_block import  DenoisingAutoencoder
# 信号分支法的处理
# 自定义一维卷积块
class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class ResNet18_1D(nn.Module):
    def __init__(self, in_channels=2):
        super(ResNet18_1D, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(16, 32, 1,stride=2)
        self.layer2 = self._make_layer(32,64, 1, stride=2)
        self.layer3 = self._make_layer(64,128, 1, stride=2)
        self.layer4 = self._make_layer(128,256, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 256)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 自定义 LSTM 模块
# class LSTMBranch(nn.Module):
#     def __init__(self, input_size=2, hidden_size=128, num_layers=2):
#         super(LSTMBranch, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_size * 2, 256)  # 双向 LSTM 的输出维度是 hidden_size * 2
#
#     def forward(self, x):
#         x, _ = self.lstm(x.permute(0, 2, 1))  # 输入形状 (batch_size, 1024, 2)
#         x = x[:, -1, :]  # 取最后一个时间步的输出
#         x = self.fc(x)
#         return x

class LSTMBranch(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):  # 减小 hidden_size 和 num_layers
        super(LSTMBranch, self).__init__()
        # 使用 GRU 替代 LSTM，减少计算量
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 256)  # 双向 GRU 的输出维度是 hidden_size * 2

    def forward(self, x):
        x, _ = self.gru(x.permute(0, 2, 1))  # 输入形状 (batch_size, 1024, 2)
        x = x[:, -1, :]  # 取最后一个时间步的输出
        x = self.fc(x)
        return x


class Signal_Branch(nn.Module):
    def __init__(self):
        super(Signal_Branch, self).__init__()
        self.resnet18 = ResNet18_1D(in_channels=2)
        # self.lstm_branch = LSTMBranch()

    def forward(self, x):
        # ResNet18 分支
        resnet_features = self.resnet18(x)  # 转换形状 (batch_size, 2, 1024) -> (batch_size, 2, 1, 1024)

        # LSTM 分支
        # lstm_features = self.lstm_branch(denoise_signal)

        # return resnet_features, lstm_features
        return resnet_features

# 图像分支的处理

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition(x)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        self.in_channels = in_channels  # Save initial input channels
        for _ in range(num_layers):
            layers.append(self._make_layer(in_channels, growth_rate))
            in_channels += growth_rate  # Increase input channels for next layer
        self.block = nn.Sequential(*layers)

    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        for layer in self.block:
            out = layer(x)
            x = torch.cat([x, out], 1)  # Concatenate input and output along the channel dimension
        return x  # Return concatenated feature maps

class Image_Branch(nn.Module):
    def __init__(self, growth_rate=24, output_features=256):
        super(Image_Branch, self).__init__()
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dense1 = DenseBlock(2 * growth_rate, growth_rate, num_layers=3)
        dense1_out_channels = 2 * growth_rate + 3 * growth_rate
        self.trans1 = TransitionLayer(dense1_out_channels, 2 * growth_rate)

        self.dense2 = DenseBlock(2 * growth_rate, growth_rate, num_layers=3)
        dense2_out_channels = 2 * growth_rate + 3 * growth_rate
        self.trans2 = TransitionLayer(dense2_out_channels, 2 * growth_rate)

        self.dense3 = DenseBlock(2 * growth_rate, growth_rate, num_layers=3)
        dense3_out_channels = 2 * growth_rate + 3 * growth_rate

        # 修改过渡层和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dense3_out_channels, output_features)  # Corrected input size

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, feature_dim))  # Attention weights

    def forward(self, signal_features, image_features):
        # 计算注意力分数
        attention_scores = torch.matmul(signal_features, self.attention_weights)
        attention_scores = torch.matmul(attention_scores, image_features.transpose(0, 1))
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        weighted_signal_features = torch.matmul(attention_weights, signal_features)
        weighted_image_features = torch.matmul(attention_weights.transpose(0, 1), image_features)

        # 融合特征
        fused_features = weighted_signal_features + weighted_image_features
        return fused_features

# 网络整体
class DAE_dual_branch(nn.Module):
    def __init__(self, num_classes):
        super(DAE_dual_branch, self).__init__()
        # 前置降噪
        # self.Denoise = DenoisingAutoencoder()
        self.Signal_Branch = Signal_Branch()

        self.Image_Branch = Image_Branch()

        # self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习的融合比例


        # 简易版融合分类
        # 融合层，包含学习比例的参数
        # self.beta = nn.Parameter(torch.tensor(0.5))  # 初始化图像分支权重
        # # 融合层
        # self.fusion_conv = nn.Conv1d(512, 256, kernel_size=1)
        # # Final classifier
        # self.fc = nn.Linear(256, num_classes)

        # 注意力机制融合分类
        # 注意力机制
        self.attention = Attention(feature_dim=256)
        # 融合层
        self.fusion_conv = nn.Conv1d(256, 256, kernel_size=1)
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        # Final classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, image):
        # 信号分支
        # denoise_signal = self.Denoise(x)
        # resnet_features, lstm_features = self.Signal_Branch(x)
        resnet_features = self.Signal_Branch(x)
        # signal_features = self.alpha * resnet_features + (1 - self.alpha) * lstm_features
        signal_features = resnet_features
        # 图像分支
        image_features = self.Image_Branch(image)

        # 打印特征的形状以调试

        # # 简易版融合与分类
        # combined_features = torch.cat((signal_features, image_features), dim=1)
        # fused_features = self.fusion_conv(combined_features)
        # output = self.fc(fused_features)

        # 注意力机制融合分类
        fused_features = self.attention(signal_features, image_features)
        # 使用池化
        fused_features = fused_features.unsqueeze(2)  # Add dummy dimension for Conv1d
        pooled_features = self.fusion_conv(fused_features)
        pooled_features = pooled_features.squeeze(2)  # Remove the dummy dimension
        # Apply dropout
        dropped_features = self.dropout(pooled_features)
        # Classification
        output = self.fc(dropped_features)

        # 根据训练和测试阶段输出不同
        if self.training:
            # 训练时输出分类结果和降噪后的信号
            # return output, denoise_signal
            return output
        else:
            # 测试时仅输出分类结果
            return output