import torch
from .attention_augmented_convolution import AugmentedConv



class micro_to_macro_model(torch.nn.Module):
    def __init__(self):
        super(micro_to_macro_model, self).__init__()

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
            in_channels=16,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=512,
        )
        self.conv_7 = AugmentedConv(
            in_channels=16,
            out_channels=16,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=256,
        )
        self.conv_6 = AugmentedConv(
            in_channels=16,
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
        
        self.fcup = torch.nn.Linear(138, 138)

    def forward(self, x):
        acnn1, idx_1 = self.maxpool(self.leaky(self.conv_1(x)))
        acnn2, idx_2 = self.maxpool(self.leaky(self.conv_2(acnn1)))
        acnn3, idx_3 = self.maxpool(self.leaky(self.conv_3(acnn2)))
        acnn4, idx_4 = self.maxpool(self.leaky(self.conv_4(acnn3)))
        acnn4_ = acnn4.reshape(acnn4.size(0), -1)
        
        up = self.leaky(self.fcup(acnn4_))
        up = up.view(acnn4.size())
        upacnn1 = self.conv_5(up)
        upsampled1 = self.maxunpool(upacnn1, idx_4, output_size=idx_3.size())
        upsampled1_ = self.conv_inbetween(upsampled1)
        
        upacnn2 = self.conv_6(upsampled1_)
        upsampled2 = self.maxunpool(upacnn2, idx_3, output_size=idx_2.size())
        upacnn3 = self.conv_7(upsampled2)
        upsampled3 = self.maxunpool(upacnn3, idx_2, output_size=idx_1.size())
        upacnn4 = self.conv_8(upsampled3)
        upsampled4 = self.maxunpool(upacnn4, idx_1, output_size=x.size())

        corr = self.conv_final(upsampled4)

        return corr



class calibration_model(torch.nn.Module):
    def __init__(self, out_layers):
        super(calibration_model, self).__init__()

        self.conv_1 = AugmentedConv(
            in_channels=1,
            out_channels=64,
            kernel_size=17,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            #             padding=5,
            shape=128,
        )
        self.conv_2 = AugmentedConv(
            in_channels=64,
            out_channels=32,
            kernel_size=11,
            dk=40,
            dv=4,
            Nh=4,
            stride=1,
            shape=64,
        )
        self.conv_3 = AugmentedConv(
            in_channels=32,
            out_channels=3,
            kernel_size=11,
            dk=20,
            dv=2,
            Nh=2,
            stride=1,
            #             padding=5,
            shape=32,
        )

        self.maxpool = torch.nn.MaxPool1d(2)
        self.leaky = torch.nn.LeakyReLU()

        self.fc1 = torch.nn.Linear(276, 128)
        self.fc2 = torch.nn.Linear(128, out_layers)

        self.bn_1 = torch.nn.BatchNorm1d(64)
        self.bn_2 = torch.nn.BatchNorm1d(32)
        self.bn_3 = torch.nn.BatchNorm1d(3)

    def forward(self, x):
        out = self.bn_1(self.maxpool(self.leaky(self.conv_1(x))))
        out = self.bn_2(self.maxpool(self.leaky(self.conv_2(out))))
        out = self.bn_3(self.maxpool(self.leaky(self.conv_3(out))))
        out = out.reshape(out.size(0), -1)
        out = self.leaky(self.fc1(out))
        out = self.fc2(out)

        return out
