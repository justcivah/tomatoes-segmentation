import torch.nn as nn
import torch

# param. count: 19,201
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        return self.out_conv(x)
    

# param. count: 190,609
class CNN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        return self.out_conv(x)
    

# param. count: 190,609
# dropping out single pixels
class DropoutCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        return self.out_conv(x)
    

# param. count: 190,609
# dropping out entire channels
class Dropout2DCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.layers(x)
        return self.out_conv(x)


# param. count: 190,609
# downsampling using maxpool
class EDPoolingCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return self.out_conv(d)
    

# param. count: 383,857
# downsampling using maxpool
# two encoders (3x3, 5x5), one decoder (3x3) 
class DoubleEDPoolingCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(48, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # this should help adding complexity to the decoder, as encoders are way much complex
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(x)
        d = self.decoder(torch.cat([e1, e2], dim=1))
        return self.out_conv(d)
    

# param. count: 190,609
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using the same kernel and layer
class EDStridingCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return self.out_conv(d)
    

# param. count: 384,593
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using different kernels and in different layers
class EDSplitStridingCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return self.out_conv(d)
    

# param. count: 941,745
# adding two different encoders, while keeping one decoder
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using different kernels and in different layers
class DoubleEDCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # this should help adding complexity to the decoder, as encoders are way much complex
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(x)
        d = self.decoder(torch.concat([e1, e2], dim=1))
        return self.out_conv(d)
    

# param. count: 987,825
# applying skip connections from encoders to decoder
# adding two different encoders, while keeping one decoder
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using different kernels and in different layers
class SkipDoubleEDCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        # encoder 2 (3x3)
        self.encoder1_step1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.encoder1_step2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.encoder1_step3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # encoder 2 (5x5)
        self.encoder2_step1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.encoder2_step2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.encoder2_step3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # decoder
        self.decoder_step1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # this should help adding complexity to the decoder, as encoders are way much complex
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step2 = nn.Sequential(
            nn.ConvTranspose2d(192, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step3 = nn.Sequential(
            nn.ConvTranspose2d(96, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1_1 = self.encoder1_step1(x)
        e1_2 = self.encoder1_step2(e1_1)
        e1_3 = self.encoder1_step3(e1_2)

        e2_1 = self.encoder2_step1(x)
        e2_2 = self.encoder2_step2(e2_1)
        e2_3 = self.encoder2_step3(e2_2)

        d_1 = self.decoder_step1(torch.concat([e1_3, e2_3], dim=1))
        d_2 = self.decoder_step2(torch.concat([d_1, e1_2, e2_2], dim=1))
        d_3 = self.decoder_step3(torch.concat([d_2, e1_1, e2_1], dim=1))
        return self.out_conv(d_3)
    

# param. count: 1,196,721
# applying cross-fusion, that is skip connections between encoders
# applying skip connections from encoders to decoder
# adding two different encoders, while keeping one decoder
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using different kernels and in different layers
class SkipBothDoubleEDCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        # encoder 2 (3x3)
        self.encoder1_step1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.encoder1_step2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.encoder1_step3 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # encoder 2 (5x5)
        self.encoder2_step1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.encoder2_step2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.encoder2_step3 = nn.Sequential(
            nn.Conv2d(128, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # decoder
        self.decoder_step1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # this should help adding complexity to the decoder, as encoders are way much complex
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step2 = nn.Sequential(
            nn.ConvTranspose2d(192, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step3 = nn.Sequential(
            nn.ConvTranspose2d(96, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1_1 = self.encoder1_step1(x)
        e2_1 = self.encoder2_step1(x)

        e1_2 = self.encoder1_step2(e1_1)
        e2_2 = self.encoder2_step2(e2_1)

        e1_3 = self.encoder1_step3(torch.concat([e1_2, e2_2], dim=1))
        e2_3 = self.encoder2_step3(torch.concat([e2_2, e1_2], dim=1))

        d_1 = self.decoder_step1(torch.concat([e1_3, e2_3], dim=1))
        d_2 = self.decoder_step2(torch.concat([d_1, e1_2, e2_2], dim=1))
        d_3 = self.decoder_step3(torch.concat([d_2, e1_1, e2_1], dim=1))
        return self.out_conv(d_3)
    

# param. count: 657,473
# also applying skip fusion, in order to keep channels number down
# applying cross-fusion, that is skip connections between encoders
# applying skip connections from encoders to decoder
# adding two different encoders, while keeping one decoder
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using different kernels and in different layers
class LightSkipBothDoubleEDCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        # encoder 2 (3x3)
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
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
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
            nn.Conv2d(96, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # decoder
        self.decoder_step1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # this should help adding complexity to the decoder, as encoders are way much complex
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step2 = nn.Sequential(
            nn.ConvTranspose2d(160, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step3 = nn.Sequential(
            nn.ConvTranspose2d(80, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        e1_1 = self.encoder1_step1(x)
        e2_1 = self.encoder2_step1(x)

        e1_2 = self.encoder1_step2(e1_1)
        e2_2 = self.encoder2_step2(e2_1)

        e1_3 = self.encoder1_step3(torch.concat([e1_2, e2_2], dim=1))
        e2_3 = self.encoder2_step3(torch.concat([e2_2, e1_2], dim=1))

        d_1 = self.decoder_step1(torch.concat([e1_3, e2_3], dim=1))
        d_2 = self.decoder_step2(torch.concat([d_1, e1_2, e2_2], dim=1))
        d_3 = self.decoder_step3(torch.concat([d_2, e1_1, e2_1], dim=1))
        return self.out_conv(d_3)
    

# param. count: 596,153
# also applying skip fusion, in order to keep channels number down
# applying cross-fusion, that is skip connections between encoders
# applying skip connections from encoders to decoder
# adding two different encoders, while keeping one decoder
# downsampling using convolutions (learned)
# here i do feature extraction and downsampling using different kernels and in different layers
class LightFuseSkipBothDoubleEDCNN(nn.Module):
    def __init__(self, in_channels=3, dropout=0.25):
        super().__init__()
        # encoder 2 (3x3)
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
            nn.Conv2d(72, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
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
            nn.Conv2d(72, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        # fuse skip 1
        self.fuse_skip_e1 = nn.Conv2d(48, 24, kernel_size=1)
        self.fuse_skip_e2 = nn.Conv2d(48, 24, kernel_size=1)
        self.fuse_skip_d1 = nn.Conv2d(96, 48, kernel_size=1)
        self.fuse_skip_d2 = nn.Conv2d(48, 24, kernel_size=1)

        # decoder
        self.decoder_step1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # this should help adding complexity to the decoder, as encoders are way much complex
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step2 = nn.Sequential(
            nn.ConvTranspose2d(112, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder_step3 = nn.Sequential(
            nn.ConvTranspose2d(56, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # encoder
        e1_1 = self.encoder1_step1(x)
        e2_1 = self.encoder2_step1(x)

        e1_2 = self.encoder1_step2(e1_1)
        e2_2 = self.encoder2_step2(e2_1)

        se_1 = self.fuse_skip_e1(e2_2)
        se_2 = self.fuse_skip_e2(e1_2)
        e1_3 = self.encoder1_step3(torch.concat([e1_2, se_2], dim=1))
        e2_3 = self.encoder2_step3(torch.concat([e2_2, se_1], dim=1))

        # decoder
        d_1 = self.decoder_step1(torch.concat([e1_3, e2_3], dim=1))

        sd_1 = self.fuse_skip_d1(torch.concat([e1_2, e2_2], dim=1))
        d_2 = self.decoder_step2(torch.concat([d_1, sd_1], dim=1))

        sd_2 = self.fuse_skip_d2(torch.concat([e1_1, e2_1], dim=1))
        d_3 = self.decoder_step3(torch.concat([d_2, sd_2], dim=1))
        return self.out_conv(d_3)
    

# TODO: to decrease the number of parameters i can cut by half (or maybe even a quarter) the channels used for skip connections. also consider lowering the overall channel count while keeping it a multiple of 2.
# TODO: in the encoders last step, is increasing to 5x5 and 7x7 better?
# TODO: for double encoder models, if image are not that sharp, it may be good adding a bit of complexity to the decoder.