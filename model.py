import torch.nn as nn
import torch

# param. count: 1,720,529
class ToNet(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        # encoder 1 (3x3)
        self.encoder1_step1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.encoder1_step2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.encoder1_step3 = nn.Sequential(
            nn.Conv2d(96, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(80, 80, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        # encoder 2 (5x5)
        self.encoder2_step1 = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Conv2d(24, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.encoder2_step2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            nn.Conv2d(48, 48, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.encoder2_step3 = nn.Sequential(
            nn.Conv2d(96, 80, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),

            nn.Conv2d(80, 80, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(160, 160, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Conv2d(160, 160, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.Dropout2d(dropout),
        )

        # decoder
        self.decoder_step1 = nn.Sequential(
            nn.ConvTranspose2d(160, 96, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Dropout2d(dropout),
        )

        self.decoder_step2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            
            nn.Dropout2d(dropout),
        )

        self.decoder_step3 = nn.Sequential(
            nn.ConvTranspose2d(144, 80, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )

        self.decoder_step4 = nn.Sequential(
            nn.Conv2d(80, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder_step5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        #encoder
        e1_1 = self.encoder1_step1(x)
        e2_1 = self.encoder2_step1(x)

        e1_2 = self.encoder1_step2(e1_1)
        e2_2 = self.encoder2_step2(e2_1)

        e1_3 = self.encoder1_step3(torch.concat([e1_2, e2_2], dim=1))
        e2_3 = self.encoder2_step3(torch.concat([e2_2, e1_2], dim=1))

        # bottleneck
        b = self.bottleneck(torch.concat([e1_3, e2_3], dim=1))

        # decoder
        d_1 = self.decoder_step1(b)
        d_2 = self.decoder_step2(torch.concat([d_1, e1_2, e2_2], dim=1))
        d_3 = self.decoder_step3(torch.concat([d_2, e1_1, e2_1], dim=1))
        d_4 = self.decoder_step4(d_3)
        d_5 = self.decoder_step5(d_4)

        return self.out_conv(d_5)