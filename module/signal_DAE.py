import torch
import torch.nn as nn
import torch.nn.functional as F

# 基本残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * 4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ResidualBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Second convolution layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut layer for matching dimensions if needed
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        # Residual path
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Shortcut path
        if self.shortcut is not None:
            residual = self.shortcut(residual)

        # Combine residual and shortcut
        out += residual
        out = F.relu(out)
        return out

# 自编码降噪器
class DenoisingAutoencoder1(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder1, self).__init__()
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

class DenoisingAutoencoder2(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder2, self).__init__()

        # Encoder: 1D Convolutional Layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=1),  # -> [batch_size, 16, 512]
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # -> [batch_size, 32, 256]
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # -> [batch_size, 64, 128]
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # -> [batch_size, 128, 64]
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # -> [batch_size, 256, 32]
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),  # -> [batch_size, 512, 16]
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layer for the latent space
        self.fc = nn.Linear(512 * 16, 512)

        # Decoder: Fully connected layer to upsample
        self.decoder_fc = nn.Sequential(
            nn.Linear(512, 512 * 16),
            nn.ReLU(True)
        )

        # Unflatten layer to reshape into convolutional shape
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 16))

        # Decoder: 1D Transposed Convolutional Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [batch_size, 256, 32]
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [batch_size, 128, 64]
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [batch_size, 64, 128]
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [batch_size, 32, 256]
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> [batch_size, 16, 512]
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1)  # -> [batch_size, 2, 1024]
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)

        # Decoding
        x = self.decoder_fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)

        return x

class DenoisingAutoencoder3(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder3, self).__init__()

        # Encoder: 1D Convolutional Layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 16, 512]
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 32, 256]
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 64, 128]
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),  # -> [batch_size, 128, 64]
            nn.ReLU(True)
        )

        # Fully connected layer for the latent space
        self.fc = nn.Linear(128 * 64, 256)

        # Decoder: Fully connected layer to upsample
        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 128 * 64),
            nn.ReLU(True)
        )

        # Decoder: 1D Transposed Convolutional Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 64, 128]
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 32, 256]
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),  # -> [batch_size, 16, 512]
            nn.ReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=5, stride=2, padding=2, output_padding=1)  # -> [batch_size, 2, 1024]
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)

        # Decoding
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 128, 64)  # Unflatten
        x = self.decoder(x)

        return x

# 多分支融合
class MutiFusion1D(nn.Module):
    def __init__(self, h_channels, m_channels, l_channels):
        super(MutiFusion1D, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # Low branch (1D Conv layers)
        self.low_1 = nn.Sequential(
            nn.Conv1d(in_channels=h_channels, out_channels=m_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(m_channels),
        )
        self.low_2 = nn.Sequential(
            nn.Conv1d(in_channels=m_channels, out_channels=l_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(l_channels),
        )

        # Mid branch (1D Conv layers)
        self.mid_1 = nn.Sequential(
            nn.Conv1d(in_channels=h_channels, out_channels=m_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(m_channels),
        )
        self.mid_2 = nn.Sequential(
            nn.Conv1d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(m_channels),
        )

        # High branch (1D Conv layers)
        self.high_1 = nn.Sequential(
            nn.Conv1d(in_channels=l_channels, out_channels=m_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(m_channels),
        )
        self.high_2 = nn.Sequential(
            nn.Conv1d(in_channels=m_channels, out_channels=h_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(h_channels),
        )

    def forward(self, h, m, l):
        # Low branch
        l1 = self.relu(self.low_1(h) + m)
        l1 = self.relu(self.low_2(l1) + l)

        # High branch with upsampling
        h1 = self.relu(F.interpolate(self.high_1(l), scale_factor=2, mode='linear', align_corners=False) + m)
        h1 = self.relu(F.interpolate(self.high_2(h1), scale_factor=2, mode='linear', align_corners=False) + h)

        # Mid branch
        m = self.relu(F.interpolate(self.mid_2(l), scale_factor=2, mode='linear', align_corners=False) + m)
        m = self.relu(self.mid_1(h) + m)

        return h1, m, l1

# 注意力机制融合
class AttentionFusionModule(nn.Module):
    def __init__(self, high_res_channels, low_res_channels):
        super(AttentionFusionModule, self).__init__()
        # 上采样低分辨率特征使其分辨率与高分辨率特征一致
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        # F.interpolate(low_res_feat, scale_factor=2, mode='linear', align_corners=False)

        # 将低分辨率特征的通道数降至与高分辨率特征一致
        self.conv_low_res = nn.Conv1d(low_res_channels, high_res_channels, kernel_size=1, stride=1)
        # 注意力机制：生成一个通道注意力权重
        self.attention = nn.Sequential(
            nn.Conv1d(high_res_channels * 2, high_res_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(high_res_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(high_res_channels, high_res_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        self.final = nn.Sequential(
            nn.Conv1d(high_res_channels * 2, high_res_channels, kernel_size=1, stride=1),
            nn.BatchNorm1d(high_res_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, high_res_feat, low_res_feat):
        # 上采样低分辨率特征
        low_res_feat_up = self.upsample(low_res_feat)
        # low_res_feat_up = self.upsample(low_res_feat, mode)

        # 通过1x1卷积将低分辨率特征的通道数缩减到与高分辨率特征一致
        low_res_feat_up = self.conv_low_res(low_res_feat_up)
        # 拼接高分辨率特征和上采样后的低分辨率特征，在通道维度上进行
        concat_feat = torch.cat([high_res_feat, low_res_feat_up], dim=1)
        # 通过注意力机制生成权重
        attention_weights = self.attention(concat_feat)
        # 使用注意力权重调整低分辨率特征的贡献

        low_res_feat_weighted = low_res_feat_up * attention_weights
        # 将加权后的低分辨率特征与高分辨率特征相加，进行融合
        low_res_feat_weighted = torch.cat([high_res_feat, low_res_feat_weighted], dim=1)
        # fused_feat = high_res_feat + low_res_feat_weighted
        fused_feat = self.final(low_res_feat_weighted)

        return fused_feat

# 模型框架
class DAE_Res_fusion(nn.Module):
    def __init__(self, num_classes=24):
        super(DAE_Res_fusion, self).__init__()
        # 降噪器
        self.dae = DenoisingAutoencoder1()
        # 处理部分
        self.stage0 = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size=1),
            nn.BatchNorm1d(8),
            ResidualBlock(8,8,7),
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            ResidualBlock(16, 16, 7),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            ResidualBlock(32, 32, 7),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            ResidualBlock(64, 64, 7),
        ) #1/8

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, 7),
            ResidualBlock(128, 128, 7),
            ResidualBlock(128, 128, 7),
        )

        self.stage2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            ResidualBlock(128, 128, 7),
            ResidualBlock(128, 128, 7),
            ResidualBlock(128, 128, 7),
        )

        self.stage31 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            ResidualBlock(256, 256, 7),
        )
        self.stage32 = ResidualBlock(256, 256, 7)
        self.stage33 = ResidualBlock(256, 256, 7)

        self.exchange1 = MutiFusion1D(128, 128, 256)
        self.exchange2 = MutiFusion1D(128, 128, 256)
        self.exchange3 = MutiFusion1D(128, 128, 256)

        self.fusion1 = AttentionFusionModule(128, 128)
        self.fusion2 = AttentionFusionModule(128, 256)

        self.d_mid = ResidualBlock(128, 128, 7)
        self.d_high = ResidualBlock(128, 128, 7)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)  # FC 256
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5
        self.fc2 = nn.Linear(256, num_classes)  # FC Numclasses

    def forward(self, x):
        # 输入x形状: (batch_size, 2, 1024)
        x = self.dae(x)

        x = self.stage0(x)
        x = self.stage1(x)
        xm = self.stage2(x)

        xl = self.stage31(xm)
        x, xm, xl = self.exchange1(x, xm, xl)
        xl = self.stage32(xl)
        x, xm, xl = self.exchange2(x, xm, xl)
        xl = self.stage33(xl)
        x, xm, xl = self.exchange3(x, xm, xl)

        xm = self.d_mid(self.fusion2(xm, xl))
        x = self.d_high(self.fusion1(x, xm))

        x = self.avg_pool(x)  # Reduces to [batch_size, 64, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]
        x = F.relu(self.fc1(x))  # FC 256
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # FC Numclasses
        # x = F.softmax(x, dim=1)  # Softmax over the classes
        return x

class DAE_Res_fusion2(nn.Module):
    def __init__(self, num_classes=24):
        super(DAE_Res_fusion2, self).__init__()
        # 降噪器
        self.dae = DenoisingAutoencoder1()
        # self.dae = DenoisingAutoencoder2()
        # self.dae = DenoisingAutoencoder3()
        # 处理部分
        self.stage0 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            ResidualBlock(64,64,7),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            ResidualBlock(64, 64, 7),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            ResidualBlock(64, 64, 7),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            ResidualBlock(64, 64, 7),
        ) #1/8

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, 7),
            ResidualBlock(128, 128, 7),
            ResidualBlock(128, 128, 7),
        )

        self.stage2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            ResidualBlock(128, 128, 7),
            ResidualBlock(128, 128, 7),
            ResidualBlock(128, 128, 7),
        )

        self.stage31 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(256),
            ResidualBlock(256, 256, 7),
        )
        self.stage32 = ResidualBlock(256, 256, 7)
        self.stage33 = ResidualBlock(256, 256, 7)

        self.exchange1 = MutiFusion1D(128, 128, 256)
        self.exchange2 = MutiFusion1D(128, 128, 256)
        self.exchange3 = MutiFusion1D(128, 128, 256)

        self.final = nn.Sequential(
            ResidualBlock(256, 256, 7),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 512)  # FC 256
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5
        self.fc2 = nn.Linear(512, num_classes)  # FC Numclasses

    def forward(self, x):
        # 输入x形状: (batch_size, 2, 1024)
        x1 = self.dae(x)

        x = self.stage0(x1)
        x = self.stage1(x)
        xm = self.stage2(x)

        xl = self.stage31(xm)
        x, xm, xl = self.exchange1(x, xm, xl)
        xl = self.stage32(xl)
        x, xm, xl = self.exchange2(x, xm, xl)
        xl = self.stage33(xl)
        x, xm, xl = self.exchange3(x, xm, xl)

        xl = self.final(xl)
        xl = self.avg_pool(xl)  # Reduces to [batch_size, 64, 1]
        xl = xl.view(xl.size(0), -1)  # Flatten to [batch_size, 64]
        xl = F.relu(self.fc1(xl))  # FC 256
        xl = self.dropout(xl)  # Apply dropout
        xl = self.fc2(xl)  # FC Numclasses
        # x = F.softmax(x, dim=1)  # Softmax over the classes
        return xl, x1

class DAE_Res_fusion_resnet(nn.Module):
    def __init__(self, num_classes=24):
        super(DAE_Res_fusion_resnet, self).__init__()
        self.in_channels = 64
        self.dae = DenoisingAutoencoder1()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer41 = self._make_layer(512, 1, stride=2)
        self.layer42 = self._make_layer(512, 1, stride=1)
        self.layer43 = self._make_layer(512, 1, stride=1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.exchange1 = MutiFusion1D(128, 256, 512)
        self.exchange2 = MutiFusion1D(128, 256, 512)

        self.att = CBAM1D(512)
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5


    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x_a = self.dae(x)
        out = F.relu(self.bn1(self.conv1(x_a)))
        out = self.layer1(out)
        out1 = self.layer2(out)
        out2 = self.layer3(out1)

        out3 = self.layer41(out2)
        out1, out2, out3 = self.exchange1(out1, out2, out3)
        out3 = self.layer42(out3)
        out1, out2, out3 = self.exchange2(out1, out2, out3)
        out3 = self.layer43(out3)

        out = self.att(out3)

        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out =self.dropout(out)
        out = self.fc2(out)
        return out, x_a



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
class ChannelAttention1D(nn.Module):
    def __init__(self, channel=256, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]
        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module(f'fc{i}', nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module(f'relu{i}', nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        # x: (batch_size, channels, length)
        res = self.avgpool(x)  # (batch_size, channels, 1)
        res = self.ca(res)  # (batch_size, channels)
        res = res.unsqueeze(-1).expand_as(x)  # (batch_size, channels, length)
        return res
class SpatialAttention1D(nn.Module):
    def __init__(self, channel=256, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1', nn.Conv1d(in_channels=channel, out_channels=channel // reduction, kernel_size=1))
        self.sa.add_module('bn_reduce1', nn.BatchNorm1d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module(f'conv_{i}', nn.Conv1d(
                in_channels=channel // reduction,
                out_channels=channel // reduction,
                kernel_size=3,
                padding=dia_val * (3 - 1) // 2,
                dilation=dia_val
            ))
            self.sa.add_module(f'bn_{i}', nn.BatchNorm1d(channel // reduction))
            self.sa.add_module(f'relu_{i}', nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv1d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        # x: (batch_size, channels, length)
        res = self.sa(x)  # (batch_size, 1, length)
        res = res.expand_as(x)  # (batch_size, channels, length)
        return res
class BAMBlock1D(nn.Module):
    def __init__(self, channel=256, reduction=16, dia_val=2):
        super().__init__()
        self.ca = ChannelAttention1D(channel=channel, reduction=reduction)
        self.sa = SpatialAttention1D(channel=channel, reduction=reduction, dia_val=dia_val)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, channels, length)
        sa_out = self.sa(x)
        ca_out = self.ca(x)
        weight = self.sigmoid(sa_out + ca_out)
        out = (1 + weight) * x
        return out


class LSTM_CNN_SAM(nn.Module):
    def __init__(self, hidden_dim=128, num_classes=24, lstm_layers=2, dropout_rate=0.2):
        super(LSTM_CNN_SAM, self).__init__()
        # 1. Fully Connected Layer
        self.stage1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            BasicBlock(64, 64, 1),
            BasicBlock(64, 64, 1),
            BasicBlock(64, 128, 2),
            BasicBlock(128, 128, 1),
            BasicBlock(128, 256, 2),
            BasicBlock(256, 256, 1),
        )
        # 2. LSTM Layer
        self.lstm = nn.LSTM(input_size=256,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        # 3. Convolution Layer
        self.conv_layer = nn.Sequential(
            BasicBlock(256, 256, 1),
            BasicBlock(256, 256, 1),
        )
        # 4. Attention Layer
        self.conv_attention = nn.Conv1d(in_channels=256,
                                        out_channels=256,
                                        kernel_size=3,
                                        padding=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.atten = BAMBlock1D()

        # 5. Fully Connected Layer for Output
        self.final = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def attention(self, x):
        # Attention Mechanism
        conv_out = self.conv_attention(x)
        attn_weights = self.softmax(self.tanh(conv_out))

        # Multiply the attention weights with the input features
        attn_out = x * attn_weights
        return attn_out

    def forward(self, x):
        x = self.stage1(x)
        aux = x

        # LSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)

        # Apply Attention Layer
        x = self.attention(x)

        x = self.atten(x)

        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)

        # Final output layer
        x = self.final(x)

        return x, aux

def main():
    # 1. 初始化模型
    # model = DAE_Res_fusion_resnet()
    model = LSTM_CNN_SAM()


    # 3. 定义输入尺寸
    input_shape = (2048, 2, 1024)
    rand_input = torch.randn(input_shape)
    # 4. 测试模型的输入输出尺寸
    # output, aux = model(rand_input)
    output = model(rand_input)
    print(f"输出尺寸: {output.shape}")

if __name__ == "__main__":
    main()
