import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import vmamba_new


class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2, mid_channels=64, out_channels=64):
        super(MultiScaleAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels1, mid_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels2, mid_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.msa = nn.MultiheadAttention(embed_dim=mid_channels, num_heads=8)
        self.output_conv = nn.Conv2d(in_channels1, out_channels, kernel_size=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.f1_ch_q = nn.Sequential(
            nn.Conv2d(in_channels1, in_channels1 // 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels1 // 2, mid_channels, 1, bias=False)
        )

    def forward(self, F1, F2):
        B, C, H1, W1 = F1.shape
        B, C, H2, W2 = F2.shape

        F1_proj = self.f1_ch_q(self.avg_pool(F1)) + self.f1_ch_q(self.max_pool(F1))

        F1_proj = self.conv1(F1_proj).view(B, -1, 1).permute(2, 0, 1)  # Shape: (H1*W1, B, C2)
        F2_proj = self.conv2(F2).view(B, -1, H2 * W2).permute(2, 0, 1)  # Shape: (H2*W2, B, C2)

        # Multi-Scale Attention
        F1_msa, _ = self.msa(F1_proj, F2_proj, F2_proj)
        F1 = self.conv1(F1).view(B, -1, H1 * W1).permute(2, 0, 1)

        F2_msa, _ = self.msa(F1, F1_msa, F1_msa)
        F2_msa = F2_msa.permute(1, 2, 0).view(B, -1, H1, W1)  # Reshape back to (B, C2, H2, W2)

        F2_out = self.output_conv(F2_msa)

        return F2_out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        return x


class VMC_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(VMC_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        resnet = models.resnet34(weights='DEFAULT')
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.bottleneck = ConvBlock(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.msa1 = MultiScaleAttention(in_channels1=64, in_channels2=512)
        self.msa2 = MultiScaleAttention(in_channels1=64, in_channels2=256)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        self.vss1 = vmamba_new.VSSLajuyer(dim=64, depth=3, drop_path=0.5)
        self.vss2 = vmamba_new.VSSLayer(dim=128, depth=3, drop_path=0.5)
        self.vss3 = vmamba_new.VSSLayer(dim=256, depth=3, drop_path=0.5)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x1 = self.inc(x)  # 64 112 112
        x2 = self.encoder1(self.pool(x1))  # 64 56 56
        x3 = self.encoder2(x2)  # 128 28 28
        x4 = self.encoder3(x3)  # 256 14 14
        x5 = self.encoder4(x4)  # 512 7 7

        x2 = self.vss1(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x3 = self.vss2(x3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x4 = self.vss3(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        dec4 = self.upconv4(x5) + x4
        dec3 = self.upconv3(dec4) + x3
        dec2 = self.upconv2(dec3) + x2
        att1 = self.msa1(dec2, x5)
        att2 = self.msa2(dec2, x4)
        dec1 = self.upconv1(dec2 + att1 + att2) + x1

        logits = self.final_conv(self.up(dec1))

        return logits


if __name__ == '__main__':
    device = torch.device("cuda")

    x = torch.randn([1, 3, 224, 224]).to(device)

    model = VMC_UNet()
    model.to(device)
    memory_before = torch.cuda.memory_allocated(device)
    segout = model(x)
    print(segout.shape)  # This should print torch.Size([1, 1, 224, 224])
