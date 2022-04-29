import torch.nn as nn
import torch.nn.functional as F
import torch
from frontend import Spectrogram


class RSM2D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(channels_out)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.bn3 = nn.BatchNorm2d(channels_out)

        self.nin = nn.Conv2d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        # xx = xx.unsqueeze(1)
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx

class SSDNet2D(nn.Module):  # 2D-Res-TSSDNet
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.RSM1 = RSM2D(channels_in=16, channels_out=32)
        self.RSM2 = RSM2D(channels_in=32, channels_out=64)
        self.RSM3 = RSM2D(channels_in=64, channels_out=128)
        self.RSM4 = RSM2D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

        self.feature = Spectrogram(320, 160, 320, 16000)

    def forward(self, x):
        x = self.feature(x)
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM3(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.RSM4(x)

        x = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        # x = F.avg_pool2d(x, kernel_size=(27, 25))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x