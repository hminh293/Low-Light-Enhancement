import torch
import torch.nn as nn
import torch.nn.functional as F

########################################
# Decomposition Network
########################################
class DecomNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 4, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        R = out[:, :3, :, :]      # Reflectance (RGB)
        L = out[:, 3:, :, :]      # Illumination (1 channel)
        return R, L


########################################
# Enhance Network (Illumination)
########################################
class EnhanceNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.enc2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.enc3 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)

        # Multi-scale feature extraction
        self.scale1 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.scale2 = nn.Conv2d(base_channels, base_channels, 5, 1, 2)
        self.scale3 = nn.Conv2d(base_channels, base_channels, 7, 1, 3)

        # Decoder
        self.dec = nn.Sequential(
            nn.Conv2d(base_channels * 3, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, L):
        x = F.relu(self.enc1(L))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))

        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)

        concat = torch.cat([s1, s2, s3], dim=1)
        L_enhanced = self.dec(concat)

        return L_enhanced


########################################
# Retinex-Net (Full Model)
########################################
class RetinexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = DecomNet()
        self.enhance = EnhanceNet()

    # =========================
    # TRAINING FORWARD
    # =========================
    def forward_train(self, x):
        """
        Dùng cho TRAINING:
        return đầy đủ để tính loss
        """
        R, L = self.decom(x)
        L_hat = self.enhance(L)
        enhanced = R * L_hat

        return enhanced, R, L, L_hat

    # =========================
    # INFERENCE / DEMO FORWARD
    # =========================
    def forward(self, x):
        """
        Dùng cho INFERENCE / STREAMLIT:
        CHỈ return ảnh enhanced RGB
        """
        R, L = self.decom(x)
        L_hat = self.enhance(L)
        enhanced = R * L_hat

        return enhanced