import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)):
        super(ResidualBlock, self).__init__()
        # 第一层卷积，使用 Conv2d，kernel_size=(1, 3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二层卷积，使用 Conv2d，kernel_size=(1, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut 层，如果输入和输出通道不一致，则使用 1x1 的卷积进行匹配
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)) if in_channels != out_channels else None

    def forward(self, x):
        # 残差路径
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Shortcut 路径
        if self.shortcut is not None:
            residual = self.shortcut(residual)

        # 合并残差和 Shortcut
        out += residual
        out = F.relu(out)
        return out

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

# 自编码降噪器
class DenoisingAutoencoder1(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder1, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        # 输入形状为 [batch_size, channels, height, width]
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DenoisingAutoencoder2(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder2, self).__init__()

        # Encoder: 2D Convolutional Layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # -> [batch_size, 16, 1, 512]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # -> [batch_size, 32, 1, 256]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # -> [batch_size, 64, 1, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # -> [batch_size, 128, 1, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # -> [batch_size, 256, 1, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),  # -> [batch_size, 512, 1, 16]
            nn.BatchNorm2d(512),
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
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 1, 16))

        # Decoder: 2D Transposed Convolutional Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # -> [batch_size, 256, 1, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # -> [batch_size, 128, 1, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # -> [batch_size, 64, 1, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # -> [batch_size, 32, 1, 256]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1)),  # -> [batch_size, 16, 1, 512]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), output_padding=(0, 1))  # -> [batch_size, 2, 1, 1024]
        )

    def forward(self, x):
        # 需要将输入形状调整为 [batch_size, channels, height, width]
        # 假设输入为 [batch_size, 2, 1024]，先调整为 [batch_size, 2, 1, 1024]
        x = x.unsqueeze(2)

        # Encoding
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc(x)

        # Decoding
        x = self.decoder_fc(x)
        x = self.unflatten(x)
        x = self.decoder(x)

        # 将形状从 [batch_size, 2, 1, 1024] 调整为 [batch_size, 2, 1024]
        x = x.squeeze(2)
        return x

class DenoisingAutoencoder3(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder3, self).__init__()

        # Encoder: 2D Convolutional Layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),  # -> [batch_size, 16, 1, 512]
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),  # -> [batch_size, 32, 1, 256]
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),  # -> [batch_size, 64, 1, 128]
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),  # -> [batch_size, 128, 1, 64]
            nn.ReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        # Fully connected layer for the latent space
        self.fc = nn.Linear(128 * 1 * 64, 256)

        # Decoder: Fully connected layer to upsample
        self.decoder_fc = nn.Sequential(
            nn.Linear(256, 128 * 1 * 64),
            nn.ReLU(True)
        )

        # Decoder: 2D Transposed Convolutional Layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1)),  # -> [batch_size, 64, 1, 128]
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1)),  # -> [batch_size, 32, 1, 256]
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1)),  # -> [batch_size, 16, 1, 512]
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), output_padding=(0, 1))  # -> [batch_size, 2, 1, 1024]
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)  # 输入为 [batch_size, 2, 1, 1024]
        x = self.flatten(x)
        x = self.fc(x)

        # Decoding
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 128, 1, 64)  # Reshape to [batch_size, 128, 1, 64]
        x = self.decoder(x)

        return x

class MutiFusion(nn.Module):
    def __init__(self, h_channels, m_channels, l_channels):
        super(MutiFusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        # Low branch (2D Conv layers)
        self.low_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.low_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=l_channels, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(l_channels),
        )

        # Mid branch (2D Conv layers)
        self.mid_1 = nn.Sequential(
            nn.Conv2d(in_channels=h_channels, out_channels=m_channels, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3), bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.mid_2 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(m_channels),
        )

        # High branch (2D Conv layers)
        self.high_1 = nn.Sequential(
            nn.Conv2d(in_channels=l_channels, out_channels=m_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(m_channels),
        )
        self.high_2 = nn.Sequential(
            nn.Conv2d(in_channels=m_channels, out_channels=h_channels, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(h_channels),
        )

    def forward(self, h, m, l):
        # Low branch
        l1 = self.relu(self.low_1(h) + m)
        l1 = self.relu(self.low_2(l1) + l)

        # High branch with upsampling
        h1 = self.relu(F.interpolate(self.high_1(l), scale_factor=(1, 2), mode='bilinear', align_corners=False) + m)
        h1 = self.relu(F.interpolate(self.high_2(h1), scale_factor=(1, 2), mode='bilinear', align_corners=False) + h)

        # Mid branch
        m = self.relu(F.interpolate(self.mid_2(l), scale_factor=(1, 2), mode='bilinear', align_corners=False) + m)
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
            nn.Conv2d(2, 64, kernel_size=(1, 7), stride=2, padding=(0, 3)),
            nn.BatchNorm2d(64),
            BasicBlock(64,64, stride=1),
            BasicBlock(64, 64, stride=2),
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=2),
        ) #1/8

        self.stage1 = nn.Sequential(
            BasicBlock(64, 128, stride=1),
            BasicBlock(128, 128, stride=1),
        )

        self.stage2 = nn.Sequential(
            BasicBlock(128, 128, stride=2),
            BasicBlock(128, 128, stride=1),
            BasicBlock(128, 128, stride=1),
        )

        self.stage31 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )

        self.stage32 = BasicBlock(256, 256, stride=1)
        self.stage33 = BasicBlock(256, 256, stride=1)

        self.exchange1 = MutiFusion(128, 128, 256)
        self.exchange2 = MutiFusion(128, 128, 256)
        self.exchange3 = MutiFusion(128, 128, 256)

        self.fusion1 = AttentionFusionModule(128, 128)
        self.fusion2 = AttentionFusionModule(128, 256)

        self.d_mid = BasicBlock(128, 128, stride=1)
        self.d_high = BasicBlock(128, 128, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 256)  # FC 256
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5
        self.fc2 = nn.Linear(256, num_classes)  # FC Numclasses

    def forward(self, x):
        # 输入x形状: (batch_size, 2, 1024)
        x_a = self.dae(x)

        x = self.stage0(x_a)
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
        return x, x_a

class DAE_Res_fusion2(nn.Module):
    def __init__(self, num_classes=24):
        super(DAE_Res_fusion2, self).__init__()
        # 降噪器
        # self.dae = DenoisingAutoencoder1()
        # self.dae = DenoisingAutoencoder2()
        self.dae = DenoisingAutoencoder3()
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

        self.exchange1 = MutiFusion(128, 128, 256)
        self.exchange2 = MutiFusion(128, 128, 256)
        self.exchange3 = MutiFusion(128, 128, 256)

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

class DAE_Res_fusion3(nn.Module):
    def __init__(self, num_classes=24):
        super(DAE_Res_fusion3, self).__init__()
        # 降噪器使用2D版本
        self.dae = DenoisingAutoencoder3()

        # 处理部分
        self.stage0 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 64, (1, 7), padding=(0, 3)),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 64, (1, 7), padding=(0, 3)),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 64, (1, 7), padding=(0, 3)),
            nn.Conv2d(64, 64, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(64),
            ResidualBlock(64, 64, (1, 7), padding=(0, 3)),
        )  # 1/8

        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, (1, 7), padding=(0, 3)),
            ResidualBlock(128, 128, (1, 7), padding=(0, 3)),
            ResidualBlock(128, 128, (1, 7), padding=(0, 3)),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(128),
            ResidualBlock(128, 128, (1, 7), padding=(0, 3)),
            ResidualBlock(128, 128, (1, 7), padding=(0, 3)),
            ResidualBlock(128, 128, (1, 7), padding=(0, 3)),
        )

        self.stage31 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 7), stride=(1, 2), padding=(0, 3)),
            nn.BatchNorm2d(256),
            ResidualBlock(256, 256, (1, 7), padding=(0, 3)),
        )
        self.stage32 = ResidualBlock(256, 256, (1, 7), padding=(0, 3))
        self.stage33 = ResidualBlock(256, 256, (1, 7), padding=(0, 3))

        # 使用2D版本的MutiFusion
        self.exchange1 = MutiFusion(128, 128, 256)
        self.exchange2 = MutiFusion(128, 128, 256)
        self.exchange3 = MutiFusion(128, 128, 256)

        self.final = nn.Sequential(
            ResidualBlock(256, 256, (1, 7), padding=(0, 3)),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 512)  # FC 256
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5
        self.fc2 = nn.Linear(512, num_classes)  # FC Numclasses

    def forward(self, x):
        # 输入x形状: (batch_size, 2, 1024)
        x = x.unsqueeze(2)  # 调整为2D格式 [batch_size, 2, 1, 1024]
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
        xl = self.avg_pool(xl)  # Reduces to [batch_size, 256, 1, 1]
        xl = xl.view(xl.size(0), -1)  # Flatten to [batch_size, 256]
        xl = F.relu(self.fc1(xl))  # FC 256
        xl = self.dropout(xl)  # Apply dropout
        xl = self.fc2(xl)  # FC Numclasses

        return xl, x1.squeeze(2)

