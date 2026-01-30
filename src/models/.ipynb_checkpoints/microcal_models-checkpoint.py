import torch
from .attention_augmented_convolution import AugmentedConv

class MicroToMacroModel(torch.nn.Module):
    def __init__(self):
        super(MicroToMacroModel, self).__init__()

        # Define attention augmented convolutional layers
        self.conv_1 = self._create_augmented_conv(1, 16, 11, 40, 4, 4, 1, 512)
        self.conv_2 = self._create_augmented_conv(16, 16, 11, 40, 4, 4, 1, 256)
        self.conv_3 = self._create_augmented_conv(16, 16, 11, 40, 4, 4, 1, 128)
        self.conv_4 = self._create_augmented_conv(16, 3, 11, 20, 2, 2, 1, 32)

        self.conv_8 = self._create_augmented_conv(16, 16, 11, 40, 4, 4, 1, 512)
        self.conv_7 = self._create_augmented_conv(16, 16, 11, 40, 4, 4, 1, 256)
        self.conv_6 = self._create_augmented_conv(16, 16, 11, 40, 4, 4, 1, 128)
        self.conv_inbetween = self._create_augmented_conv(3, 16, 11, 20, 2, 2, 1, 32)
        self.conv_5 = self._create_augmented_conv(3, 3, 11, 20, 2, 2, 1, 32)

        # Final convolution layer
        self.conv_final = torch.nn.Conv1d(in_channels=16, out_channels=1, kernel_size=5, padding=2)

        # Pooling and activation layers
        self.maxpool = torch.nn.MaxPool1d(2, return_indices=True)
        self.maxunpool = torch.nn.MaxUnpool1d(2)
        self.leaky = torch.nn.LeakyReLU()

        # Fully connected layer
        self.fcup = torch.nn.Linear(138, 138)

    def _create_augmented_conv(self, in_channels, out_channels, kernel_size, dk, dv, Nh, stride, shape):
        return AugmentedConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dk=dk,
            dv=dv,
            Nh=Nh,
            stride=stride,
            shape=shape,
        )

    def forward(self, x):
        # Encoder 
        acnn1, idx_1 = self.maxpool(self.leaky(self.conv_1(x)))
        acnn2, idx_2 = self.maxpool(self.leaky(self.conv_2(acnn1)))
        acnn3, idx_3 = self.maxpool(self.leaky(self.conv_3(acnn2)))
        acnn4, idx_4 = self.maxpool(self.leaky(self.conv_4(acnn3)))

        # Fully connected layer
        acnn4_ = acnn4.reshape(acnn4.size(0), -1)
        up = self.leaky(self.fcup(acnn4_))
        up = up.view(acnn4.size())

        # Decoder 
        upacnn1 = self.conv_5(up)
        upsampled1 = self.maxunpool(upacnn1, idx_4, output_size=idx_3.size())
        upsampled1_ = self.conv_inbetween(upsampled1)

        upacnn2 = self.conv_6(upsampled1_)
        upsampled2 = self.maxunpool(upacnn2, idx_3, output_size=idx_2.size())
        upacnn3 = self.conv_7(upsampled2)
        upsampled3 = self.maxunpool(upacnn3, idx_2, output_size=idx_1.size())
        upacnn4 = self.conv_8(upsampled3)
        upsampled4 = self.maxunpool(upacnn4, idx_1, output_size=x.size())

        # Final convolution
        corr = self.conv_final(upsampled4)
        return corr


class CalibrationModel(torch.nn.Module):
    def __init__(self, out_layers):
        super(CalibrationModel, self).__init__()

        # Define attention augmented convolutional layers with batch normalization
        self.conv_1 = self._create_augmented_conv(1, 64, 17, 40, 4, 4, 1, 128)
        self.bn_1 = torch.nn.BatchNorm1d(64)

        self.conv_2 = self._create_augmented_conv(64, 32, 11, 40, 4, 4, 1, 64)
        self.bn_2 = torch.nn.BatchNorm1d(32)

        self.conv_3 = self._create_augmented_conv(32, 3, 11, 20, 2, 2, 1, 32)
        self.bn_3 = torch.nn.BatchNorm1d(3)

        # Pooling and activation layers
        self.maxpool = torch.nn.MaxPool1d(2)
        self.leaky = torch.nn.LeakyReLU()

        # Fully connected layers
        self.fc1 = torch.nn.Linear(276, 128)
        self.fc2 = torch.nn.Linear(128, out_layers)

    def _create_augmented_conv(self, in_channels, out_channels, kernel_size, dk, dv, Nh, stride, shape):
        return AugmentedConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dk=dk,
            dv=dv,
            Nh=Nh,
            stride=stride,
            shape=shape,
        )

    def forward(self, x):
        # Convolution with batch normalization and pooling
        out = self.bn_1(self.leaky(self.conv_1(x)))
        out = self.maxpool(out)

        out = self.bn_2(self.leaky(self.conv_2(out)))
        out = self.maxpool(out)

        out = self.bn_3(self.leaky(self.conv_3(out)))
        out = self.maxpool(out)

        # Flatten 
        out = out.reshape(out.size(0), -1)

        # Fully connected layers
        out = self.leaky(self.fc1(out))
        out = self.fc2(out)

        return out