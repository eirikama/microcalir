import torch
from attention_augmented_convolution import AugmentedConv


class attention_CNN_correction(torch.nn.Module):
    def __init__(self, skips=True):
        super(attention_CNN_correction, self).__init__()

        self.skips = skips
        if self.skips == True:
            skip_in = 32
        else:
            skip_in = 16

        self.conv_1 = AugmentedConv(
            in_channels=1,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=512,
        )
        self.conv_2 = AugmentedConv(
            in_channels=16,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=256,
        )
        self.conv_3 = AugmentedConv(
            in_channels=16,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=128,
        )
        self.conv_4 = AugmentedConv(
            in_channels=16,
            out_channels=3,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_8 = AugmentedConv(
            in_channels=skip_in,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=512,
        )
        self.conv_7 = AugmentedConv(
            in_channels=skip_in,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=256,
        )
        self.conv_6 = AugmentedConv(
            in_channels=skip_in,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=128,
        )

        self.conv_inbetween = AugmentedConv(
            in_channels=3,
            out_channels=16,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )
        self.conv_5 = AugmentedConv(
            in_channels=3,
            out_channels=3,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            shape=32,
        )

        self.conv_final = torch.nn.Conv1d(
            in_channels=16, out_channels=1, kernel_size=5, padding=2
        )

        self.maxpool = torch.nn.MaxPool1d(2, return_indices=True)
        self.maxunpool = torch.nn.MaxUnpool1d(2)
        self.leaky = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()

        self.fcup = torch.nn.Linear(138, 138)

    def forward(self, x):
        acnn1, idx_1 = self.maxpool(self.leaky(self.conv_1(x)))
        acnn2, idx_2 = self.maxpool(self.leaky(self.conv_2(acnn1)))
        acnn3, idx_3 = self.maxpool(self.leaky(self.conv_3(acnn2)))
        acnn4, idx_4 = self.maxpool(self.leaky(self.conv_4(acnn3)))

        acnn4_ = acnn4.reshape(acnn4.size(0), -1)

        # Correct Spectrum

        up = self.leaky(self.fcup(acnn4_))
        up = up.view(acnn4.size())

        if self.skips == True:
            skipped1 = torch.cat([acnn4, up], dim=1)

        if self.skips == True:
            upacnn1 = self.conv_5(skipped1)
        else:
            upacnn1 = self.conv_5(up)

        upsampled1 = self.maxunpool(upacnn1, idx_4, output_size=idx_3.size())

        upsampled1_ = self.conv_inbetween(upsampled1)

        if self.skips == True:
            skipped2 = torch.cat([acnn3, upsampled1_], dim=1)

        if self.skips == True:
            upacnn2 = self.conv_6(skipped2)
        else:
            upacnn2 = self.conv_6(upsampled1_)

        upsampled2 = self.maxunpool(upacnn2, idx_3, output_size=idx_2.size())

        if self.skips == True:
            skipped3 = torch.cat([acnn2, upsampled2], dim=1)

        if self.skips == True:
            upacnn3 = self.conv_7(skipped3)
        else:
            upacnn3 = self.conv_7(upsampled2)

        upsampled3 = self.maxunpool(upacnn3, idx_2, output_size=idx_1.size())

        if self.skips == True:
            skipped4 = torch.cat([acnn1, upsampled3], dim=1)

        if self.skips == True:
            upacnn4 = self.conv_8(skipped4)
        else:
            upacnn4 = self.conv_8(upsampled3)

        upsampled4 = self.maxunpool(upacnn4, idx_1, output_size=x.size())

        corr = self.conv_final(upsampled4)

        return corr
