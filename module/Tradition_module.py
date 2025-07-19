import torch
import torch.nn as nn
import torch.nn.functional as F


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

class ResNet18(nn.Module):
    def __init__(self, num_classes=24):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, num_classes=24):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.fc = nn.Linear(512 * 4, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class VGG(nn.Module):
    def __init__(self, num_classes=24):
        super(VGG, self).__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layers(self):
        layers = []
        in_channels = 2
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv1d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm1d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = F.adaptive_avg_pool1d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class CNN(nn.Module):
    def __init__(self, num_classes=24):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256 * 128, 512)  # 注意这里的线性层输入尺寸调整
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.size(0), -1)  # 展平
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

class L_CNN(nn.Module):
    def __init__(self, num_classes=24):
        super(L_CNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv1d(2, 8, kernel_size=7, padding=3)  # conv 7 8
        self.conv2 = nn.Conv1d(8, 16, kernel_size=7, padding=3)  # conv 7 16
        self.conv3 = nn.Conv1d(16, 32, kernel_size=7, padding=3)  # conv 7 32
        self.conv4 = nn.Conv1d(32, 64, kernel_size=7, padding=3)  # conv 7 64

        # Average pooling layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Fully connected layers
        self.fc1 = nn.Linear(64, 256)  # FC 256
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5
        self.fc2 = nn.Linear(256, num_classes)  # FC Numclasses

    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))  # conv 7 8
        x = F.relu(self.conv2(x))  # conv 7 16
        x = F.relu(self.conv3(x))  # conv 7 32
        x = F.relu(self.conv4(x))  # conv 7 64

        # Global average pooling
        x = self.avg_pool(x)  # Reduces to [batch_size, 64, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))  # FC 256
        x = self.dropout(x)  # Apply dropout

        # Final fully connected layer with softmax
        x = self.fc2(x)  # FC Numclasses
        x = F.softmax(x, dim=1)  # Softmax over the classes
        return x

# LR_CNN
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
class LR_CNN(nn.Module):
    def __init__(self, num_classes=24):
        super(LR_CNN, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Conv1d(2, 8, kernel_size=1)  # conv 1 8
        self.initial_bn = nn.BatchNorm1d(8)

        # Residual blocks
        self.res_block1 = ResidualBlock(8, 8)  # conv 7 8 x 2
        self.transition1 = nn.Conv1d(8, 16, kernel_size=1)  # conv 1 16
        self.trans_bn1 = nn.BatchNorm1d(16)

        self.res_block2 = ResidualBlock(16, 16)  # conv 7 16 x 2
        self.transition2 = nn.Conv1d(16, 32, kernel_size=1)  # conv 1 32
        self.trans_bn2 = nn.BatchNorm1d(32)

        self.res_block3 = ResidualBlock(32, 32)  # conv 7 32 x 2
        self.transition3 = nn.Conv1d(32, 64, kernel_size=1)  # conv 1 64
        self.trans_bn3 = nn.BatchNorm1d(64)

        self.res_block4 = ResidualBlock(64, 64)  # conv 7 64 x 2

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Fully connected layers
        self.fc1 = nn.Linear(64, 256)  # FC 256
        self.dropout = nn.Dropout(0.5)  # Dropout 0.5
        self.fc2 = nn.Linear(256, num_classes)  # FC Numclasses

    def forward(self, x):
        # Initial convolution and ReLU
        x = F.relu(self.initial_bn(self.initial_conv(x)))  # conv 1 8

        # Residual block 1
        x = self.res_block1(x)  # conv 7 8 -> conv 7 8 + residual
        x = F.relu(self.trans_bn1(self.transition1(x)))  # conv 1 16

        # Residual block 2
        x = self.res_block2(x)  # conv 7 16 -> conv 7 16 + residual
        x = F.relu(self.trans_bn2(self.transition2(x)))  # conv 1 32

        # Residual block 3
        x = self.res_block3(x)  # conv 7 32 -> conv 7 32 + residual
        x = F.relu(self.trans_bn3(self.transition3(x)))  # conv 1 64

        # Residual block 4
        x = self.res_block4(x)  # conv 7 64 -> conv 7 64 + residual

        # Global average pooling
        x = self.avg_pool(x)  # Reduces to [batch_size, 64, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))  # FC 256
        x = self.dropout(x)  # Apply dropout

        # Final fully connected layer with softmax
        x = self.fc2(x)  # FC Numclasses
        x = F.softmax(x, dim=1)  # Softmax over the classes
        return x

# Dual_LRCNN
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock2D, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut layer for matching dimensions if needed
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

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
class LR_CNN_dual(nn.Module):
    def __init__(self, num_classes=24):
        super(LR_CNN_dual, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Conv1d(2, 8, kernel_size=1)  # conv 1 8
        self.initial_bn = nn.BatchNorm1d(8)

        # Residual blocks
        self.res_block1 = ResidualBlock(8, 8)  # conv 7 8 x 2
        self.transition1 = nn.Conv1d(8, 16, kernel_size=1)  # conv 1 16
        self.trans_bn1 = nn.BatchNorm1d(16)

        self.res_block2 = ResidualBlock(16, 16)  # conv 7 16 x 2
        self.transition2 = nn.Conv1d(16, 32, kernel_size=1)  # conv 1 32
        self.trans_bn2 = nn.BatchNorm1d(32)

        self.res_block3 = ResidualBlock(32, 32)  # conv 7 32 x 2
        self.transition3 = nn.Conv1d(32, 64, kernel_size=1)  # conv 1 64
        self.trans_bn3 = nn.BatchNorm1d(64)

        self.res_block4 = ResidualBlock(64, 64)  # conv 7 64 x 2

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Fully connected layers
        self.fc1 = nn.Linear(64, 256)  # FC 256
        self.dropout = nn.Dropout(0.1)  # Dropout 0.5
        # self.fc2 = nn.Linear(256, num_classes)  # FC Numclasses

    def forward(self, x):
        # Initial convolution and ReLU
        x = F.relu(self.initial_bn(self.initial_conv(x)))  # conv 1 8

        # Residual block 1
        x = self.res_block1(x)  # conv 7 8 -> conv 7 8 + residual
        x = F.relu(self.trans_bn1(self.transition1(x)))  # conv 1 16

        # Residual block 2
        x = self.res_block2(x)  # conv 7 16 -> conv 7 16 + residual
        x = F.relu(self.trans_bn2(self.transition2(x)))  # conv 1 32

        # Residual block 3
        x = self.res_block3(x)  # conv 7 32 -> conv 7 32 + residual
        x = F.relu(self.trans_bn3(self.transition3(x)))  # conv 1 64

        # Residual block 4
        x = self.res_block4(x)  # conv 7 64 -> conv 7 64 + residual

        # Global average pooling
        x = self.avg_pool(x)  # Reduces to [batch_size, 64, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))  # FC 256
        x = self.dropout(x)  # Apply dropout

        # Final fully connected layer with softmax
        # x = self.fc2(x)  # FC Numclasses
        # x = F.softmax(x, dim=1)  # Softmax over the classes
        return x
class LR_CNN2D(nn.Module):
    def __init__(self, num_classes=24):
        super(LR_CNN2D, self).__init__()

        # Initial convolution for 3-channel input (e.g., RGB images)
        self.initial_conv = nn.Conv2d(3, 8, kernel_size=1)  # conv 1 8
        self.initial_bn = nn.BatchNorm2d(8)

        # Residual blocks
        self.res_block1 = ResidualBlock2D(8, 8)  # conv 3x3 8 x 2
        self.transition1 = nn.Conv2d(8, 16, kernel_size=1)  # conv 1x1 16
        self.trans_bn1 = nn.BatchNorm2d(16)

        self.res_block2 = ResidualBlock2D(16, 16)  # conv 3x3 16 x 2
        self.transition2 = nn.Conv2d(16, 32, kernel_size=1)  # conv 1x1 32
        self.trans_bn2 = nn.BatchNorm2d(32)

        self.res_block3 = ResidualBlock2D(32, 32)  # conv 3x3 32 x 2
        self.transition3 = nn.Conv2d(32, 64, kernel_size=1)  # conv 1x1 64
        self.trans_bn3 = nn.BatchNorm2d(64)

        self.res_block4 = ResidualBlock2D(64, 64)  # conv 3x3 64 x 2

        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        # Fully connected layers
        self.fc1 = nn.Linear(64, 256)  # FC 256
        self.dropout = nn.Dropout(0.1)  # Dropout 0.5
        # self.fc2 = nn.Linear(256, num_classes)  # FC Numclasses

    def forward(self, x):
        # Initial convolution and ReLU
        x = F.relu(self.initial_bn(self.initial_conv(x)))  # conv 1x1 8

        # Residual block 1
        x = self.res_block1(x)  # conv 3x3 8 -> conv 3x3 8 + residual
        x = F.relu(self.trans_bn1(self.transition1(x)))  # conv 1x1 16

        # Residual block 2
        x = self.res_block2(x)  # conv 3x3 16 -> conv 3x3 16 + residual
        x = F.relu(self.trans_bn2(self.transition2(x)))  # conv 1x1 32

        # Residual block 3
        x = self.res_block3(x)  # conv 3x3 32 -> conv 3x3 32 + residual
        x = F.relu(self.trans_bn3(self.transition3(x)))  # conv 1x1 64

        # Residual block 4
        x = self.res_block4(x)  # conv 3x3 64 -> conv 3x3 64 + residual

        # Global average pooling
        x = self.avg_pool(x)  # Reduces to [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 64]

        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))  # FC 256
        x = self.dropout(x)  # Apply dropout

        # Final fully connected layer with softmax
        # x = self.fc2(x)  # FC Numclasses
        # x = F.softmax(x, dim=1)  # Softmax over the classes
        return x

class Dual_LRCNN(nn.Module):
    def __init__(self, num_classes=24):
        super(Dual_LRCNN, self).__init__()
        self.signal = LR_CNN_dual()
        self.image = LR_CNN2D()
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)  # Output: 256
        )


    def forward(self, signal, image):
        signal_features = self.signal(signal);
        img_features = self.image(image);

        # Concatenate features from both branches
        fused_features = torch.cat((signal_features, img_features), dim=1)  # Output shape: (batch_size, 512)

        x = self.fc(fused_features);
        x = F.softmax(x, dim=1)  # Softmax over the classes
        return x

