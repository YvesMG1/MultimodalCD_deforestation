
# Importing necessary libraries
import sys
from torch.nn.modules.padding import ReplicationPad2d
import torch
import torch.nn as nn
import torch.nn.functional as F



class Unet(nn.Module):
    """
    EF-CF Model by Daudt, R. C., B. Le Saux, and A. Boulch (2018). 
    Fully convolutional siamese networks for change detection. 
    In 2018 25th IEEE international conference on image processing (ICIP), pp. 4063–4067. IEEE.
    """

    def __init__(self, input_nbr, label_nbr, kernel_size=3, dropout=0.2):
        super(Unet, self).__init__()

        self.input_nbr = input_nbr
        padding = kernel_size // 2

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=kernel_size, padding=padding)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=dropout)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=kernel_size, padding=padding)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=dropout)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=dropout)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=dropout)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=dropout)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=dropout)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=dropout)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=dropout)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=dropout)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=dropout)


        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, padding=padding)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=dropout)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=dropout)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=dropout)

        self.upconv3 = nn.ConvTranspose2d(64, 64,kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=dropout)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=dropout)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=dropout)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=dropout)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=dropout)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=dropout)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=kernel_size, padding=padding)


    def forward(self, x):


        #Forward method
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43, kernel_size=2, stride=2)


        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43.size(3) - x4d.size(3), 0, x43.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33.size(3) - x3d.size(3), 0, x33.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22.size(3) - x2d.size(3), 0, x22.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12.size(3) - x1d.size(3), 0, x12.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d


class FCSiamDiff(nn.Module):
    """
    FCSiamDiff model by Daudt, R. C., B. Le Saux, and A. Boulch (2018). 
    Fully convolutional siamese networks for change detection. 
    In 2018 25th IEEE international conference on image processing (ICIP), pp. 4063–4067. IEEE.
    """

    def __init__(self, input_nbr, label_nbr, dropout=0.5, kernel_size=3):
        super(FCSiamDiff, self).__init__()

        self.input_nbr = input_nbr
        padding = kernel_size // 2 

        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=kernel_size, padding=padding)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=dropout)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=kernel_size, padding=padding)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=dropout)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=dropout)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=dropout)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=dropout)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=dropout)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=dropout)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=dropout)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=dropout)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=dropout)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, padding=padding)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=dropout)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=dropout)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=dropout)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=dropout)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=dropout)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=dropout)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=dropout)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=dropout)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=dropout)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=kernel_size, padding=padding)


    def forward(self, x1_S2=None, x2_S2=None, x1_S1=None, x2_S1=None):

        if x1_S1 is not None and x1_S2 is not None:
            x1 = torch.cat((x1_S2, x1_S1), 1)
            x2 = torch.cat((x2_S2, x2_S1), 1)
        elif x1_S1 is not None and x1_S2 is None:
            x1 = x1_S1
            x2 = x2_S1
        elif x1_S1 is None and x1_S2 is not None:
            x1 = x1_S2
            x2 = x2_S2
        else:
            raise ValueError("No input data provided")

        """Forward method."""
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)

        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)



        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d



class FCSiamConc(nn.Module):
    """
    FCSiamConc model by Daudt, R. C., B. Le Saux, and A. Boulch (2018). 
    Fully convolutional siamese networks for change detection. 
    In 2018 25th IEEE international conference on image processing (ICIP), pp. 4063–4067. IEEE.
    """

    def __init__(self, input_nbr, label_nbr, dropout=0.2, kernel_size=3):
        super(FCSiamConc, self).__init__()

        self.input_nbr = input_nbr
        padding = kernel_size // 2
    
        self.conv11 = nn.Conv2d(input_nbr, 16, kernel_size=kernel_size, padding=padding)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=dropout)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=kernel_size, padding=padding)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=dropout)

        self.conv21 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=padding)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=dropout)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=kernel_size, padding=padding)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=dropout)

        self.conv31 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=dropout)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=dropout)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=dropout)

        self.conv41 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=dropout)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=dropout)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=dropout)

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(384, 128, kernel_size=kernel_size, padding=padding)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=dropout)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=kernel_size, padding=padding)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=dropout)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=padding)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=dropout)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(192, 64, kernel_size=kernel_size, padding=padding)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=dropout)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=kernel_size, padding=padding)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=dropout)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=padding)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=dropout)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(96, 32, kernel_size=kernel_size, padding=padding)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=dropout)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=kernel_size, padding=padding)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=dropout)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=kernel_size, padding=padding, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(48, 16, kernel_size=kernel_size, padding=padding)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=dropout)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=kernel_size, padding=padding)

        self.sm = nn.LogSoftmax(dim=1)


    def forward(self, x1_S2=None, x2_S2=None, x1_S1=None, x2_S1=None):

        if x1_S1 is not None and x1_S2 is not None:
            x1 = torch.cat((x1_S2, x1_S1), 1)
            x2 = torch.cat((x2_S2, x2_S1), 1)
        elif x1_S1 is not None and x1_S2 is None:
            x1 = x1_S1
            x2 = x2_S1
        elif x1_S1 is None and x1_S2 is not None:
            x1 = x1_S2
            x2 = x2_S2
        else:
            raise ValueError("No input data provided")


        #Forward method
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)


        ####################################################
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(x2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)


        ####################################################
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d
    


class SiamUnet_conc_multi(nn.Module):
    """
    Ebel, P., S. Saha, and X. X. Zhu (2021). Fusing multi-modal data for supervised change
    detection. The international archives of the photogrammetry, remote sensing and spatial
    information sciences 43, 243–249.
    """

    def __init__(self, input_nbr_S2, input_nbr_S1, label_nbr, dropout=0.2, kernel_size=3):
        super(SiamUnet_conc_multi, self).__init__()

        self.input_nbr_1, self.input_nbr_2 = input_nbr_S2, input_nbr_S1

        ################################# encoder S2 #################################

        # 16 channels
        self.conv11 = nn.Conv2d(self.input_nbr_1, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        # 32 channels
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        # 64 channels
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        # 128 channels
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        ################################# encoder S1 #################################

        # 16 channels
        self.conv11_b = nn.Conv2d(self.input_nbr_2, 16, kernel_size=3, padding=1)
        self.bn11_b = nn.BatchNorm2d(16)
        self.do11_b = nn.Dropout2d(p=0.2)
        self.conv12_b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12_b = nn.BatchNorm2d(16)
        self.do12_b = nn.Dropout2d(p=0.2)

        # 32 channels
        self.conv21_b = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21_b = nn.BatchNorm2d(32)
        self.do21_b = nn.Dropout2d(p=0.2)
        self.conv22_b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22_b = nn.BatchNorm2d(32)
        self.do22_b = nn.Dropout2d(p=0.2)

        # 64 channels
        self.conv31_b = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31_b = nn.BatchNorm2d(64)
        self.do31_b = nn.Dropout2d(p=0.2)
        self.conv32_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32_b = nn.BatchNorm2d(64)
        self.do32_b = nn.Dropout2d(p=0.2)
        self.conv33_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33_b = nn.BatchNorm2d(64)
        self.do33_b = nn.Dropout2d(p=0.2)

        # 128 channels
        self.conv41_b = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41_b = nn.BatchNorm2d(128)
        self.do41_b = nn.Dropout2d(p=0.2)
        self.conv42_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42_b = nn.BatchNorm2d(128)
        self.do42_b = nn.Dropout2d(p=0.2)
        self.conv43_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43_b = nn.BatchNorm2d(128)
        self.do43_b = nn.Dropout2d(p=0.2)

        ################################# decoder #################################

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(384+128+128, 128, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(192+64+64, 64, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(96+32+32, 32, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(48+16+16, 16, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)


    def forward(self, s2_1, s2_2, s1_1, s1_2):

        """Forward method."""

        #################################################### encoder S2 ####################################################

        # siamese processing of input s2_1
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(s2_1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)


        ####################################################

        # siamese processing of input s2_2
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(s2_2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        #################################################### encoder S1 ####################################################

        # siamese processing of input s1_1
        # Stage 1
        x11_b = self.do11_b(F.relu(self.bn11_b(self.conv11_b(s1_1))))
        x12_1_b = self.do12_b(F.relu(self.bn12_b(self.conv12_b(x11_b))))
        x1p_b = F.max_pool2d(x12_1_b, kernel_size=2, stride=2)


        # Stage 2
        x21_b = self.do21_b(F.relu(self.bn21_b(self.conv21_b(x1p_b))))
        x22_1_b = self.do22_b(F.relu(self.bn22_b(self.conv22_b(x21_b))))
        x2p_b = F.max_pool2d(x22_1_b, kernel_size=2, stride=2)

        # Stage 3
        x31_b = self.do31_b(F.relu(self.bn31_b(self.conv31_b(x2p_b))))
        x32_b = self.do32_b(F.relu(self.bn32_b(self.conv32_b(x31_b))))
        x33_1_b = self.do33_b(F.relu(self.bn33_b(self.conv33_b(x32_b))))
        x3p_b = F.max_pool2d(x33_1_b, kernel_size=2, stride=2)

        # Stage 4
        x41_b = self.do41_b(F.relu(self.bn41_b(self.conv41_b(x3p_b))))
        x42_b = self.do42_b(F.relu(self.bn42_b(self.conv42_b(x41_b))))
        x43_1_b = self.do43_b(F.relu(self.bn43_b(self.conv43_b(x42_b))))
        x4p_b = F.max_pool2d(x43_1_b, kernel_size=2, stride=2)


        ####################################################

        # siamese processing of input s1_2
        # Stage 1
        x11_b = self.do11_b(F.relu(self.bn11_b(self.conv11_b(s1_2))))
        x12_2_b = self.do12_b(F.relu(self.bn12_b(self.conv12_b(x11_b))))
        x1p_b = F.max_pool2d(x12_2_b, kernel_size=2, stride=2)

        # Stage 2
        x21_b = self.do21_b(F.relu(self.bn21_b(self.conv21_b(x1p_b))))
        x22_2_b = self.do22_b(F.relu(self.bn22_b(self.conv22_b(x21_b))))
        x2p_b = F.max_pool2d(x22_2_b, kernel_size=2, stride=2)

        # Stage 3
        x31_b = self.do31_b(F.relu(self.bn31_b(self.conv31_b(x2p_b))))
        x32_b = self.do32_b(F.relu(self.bn32_b(self.conv32_b(x31_b))))
        x33_2_b = self.do33_b(F.relu(self.bn33_b(self.conv33_b(x32_b))))
        x3p_b = F.max_pool2d(x33_2_b, kernel_size=2, stride=2)

        # Stage 4
        x41_b = self.do41_b(F.relu(self.bn41_b(self.conv41_b(x3p_b))))
        x42_b = self.do42_b(F.relu(self.bn42_b(self.conv42_b(x41_b))))
        x43_2_b = self.do43_b(F.relu(self.bn43_b(self.conv43_b(x42_b))))
        x4p_b = F.max_pool2d(x43_2_b, kernel_size=2, stride=2)

        #################################################### decoder ####################################################
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), x43_1, x43_2, x43_1_b, x43_2_b), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), x33_1, x33_2, x33_1_b, x33_2_b), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), x22_1, x22_2, x22_1_b, x22_2_b), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), x12_1, x12_2, x12_1_b, x12_2_b), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d



class SiamUnet_diff_multi(nn.Module):

    def __init__(self, input_nbr_S2, input_nbr_S1, label_nbr, dropout=0.2, kernel_size=3):
        super(SiamUnet_diff_multi, self).__init__()

        self.input_nbr_1, self.input_nbr_2 = input_nbr_S2, input_nbr_S1

        ################################# encoder S2 #################################

        # 16 channels
        self.conv11 = nn.Conv2d(self.input_nbr_1, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        # 32 channels
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        # 64 channels
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        # 128 channels
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        ################################# encoder S1 #################################

        # 16 channels
        self.conv11_b = nn.Conv2d(self.input_nbr_2, 16, kernel_size=3, padding=1)
        self.bn11_b = nn.BatchNorm2d(16)
        self.do11_b = nn.Dropout2d(p=0.2)
        self.conv12_b = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12_b = nn.BatchNorm2d(16)
        self.do12_b = nn.Dropout2d(p=0.2)

        # 32 channels
        self.conv21_b = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21_b = nn.BatchNorm2d(32)
        self.do21_b = nn.Dropout2d(p=0.2)
        self.conv22_b = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22_b = nn.BatchNorm2d(32)
        self.do22_b = nn.Dropout2d(p=0.2)

        # 64 channels
        self.conv31_b = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31_b = nn.BatchNorm2d(64)
        self.do31_b = nn.Dropout2d(p=0.2)
        self.conv32_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32_b = nn.BatchNorm2d(64)
        self.do32_b = nn.Dropout2d(p=0.2)
        self.conv33_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33_b = nn.BatchNorm2d(64)
        self.do33_b = nn.Dropout2d(p=0.2)

        # 128 channels
        self.conv41_b = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41_b = nn.BatchNorm2d(128)
        self.do41_b = nn.Dropout2d(p=0.2)
        self.conv42_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42_b = nn.BatchNorm2d(128)
        self.do42_b = nn.Dropout2d(p=0.2)
        self.conv43_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43_b = nn.BatchNorm2d(128)
        self.do43_b = nn.Dropout2d(p=0.2)

        ################################# decoder #################################

        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv43d = nn.ConvTranspose2d(384, 128, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv33d = nn.ConvTranspose2d(192, 64, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv22d = nn.ConvTranspose2d(96, 32, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1, stride=2, output_padding=1)

        self.conv12d = nn.ConvTranspose2d(48, 16, kernel_size=3, padding=1)  # added S1+S2 channels here
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, label_nbr, kernel_size=3, padding=1)


    def forward(self, s2_1, s2_2, s1_1, s1_2):

        """Forward method."""

        #################################################### encoder S2 ####################################################

        # siamese processing of input s2_1
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(s2_1))))
        x12_1 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_1, kernel_size=2, stride=2)


        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_1 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_1, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_1 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_1, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_1 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_1, kernel_size=2, stride=2)


        ####################################################

        # siamese processing of input s2_2
        # Stage 1
        x11 = self.do11(F.relu(self.bn11(self.conv11(s2_2))))
        x12_2 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12_2, kernel_size=2, stride=2)

        # Stage 2
        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22_2 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22_2, kernel_size=2, stride=2)

        # Stage 3
        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33_2 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33_2, kernel_size=2, stride=2)

        # Stage 4
        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43_2 = self.do43(F.relu(self.bn43(self.conv43(x42))))
        x4p = F.max_pool2d(x43_2, kernel_size=2, stride=2)

        #################################################### encoder S1 ####################################################

        # siamese processing of input s1_1
        # Stage 1
        x11_b = self.do11_b(F.relu(self.bn11_b(self.conv11_b(s1_1))))
        x12_1_b = self.do12_b(F.relu(self.bn12_b(self.conv12_b(x11_b))))
        x1p_b = F.max_pool2d(x12_1_b, kernel_size=2, stride=2)


        # Stage 2
        x21_b = self.do21_b(F.relu(self.bn21_b(self.conv21_b(x1p_b))))
        x22_1_b = self.do22_b(F.relu(self.bn22_b(self.conv22_b(x21_b))))
        x2p_b = F.max_pool2d(x22_1_b, kernel_size=2, stride=2)

        # Stage 3
        x31_b = self.do31_b(F.relu(self.bn31_b(self.conv31_b(x2p_b))))
        x32_b = self.do32_b(F.relu(self.bn32_b(self.conv32_b(x31_b))))
        x33_1_b = self.do33_b(F.relu(self.bn33_b(self.conv33_b(x32_b))))
        x3p_b = F.max_pool2d(x33_1_b, kernel_size=2, stride=2)

        # Stage 4
        x41_b = self.do41_b(F.relu(self.bn41_b(self.conv41_b(x3p_b))))
        x42_b = self.do42_b(F.relu(self.bn42_b(self.conv42_b(x41_b))))
        x43_1_b = self.do43_b(F.relu(self.bn43_b(self.conv43_b(x42_b))))
        x4p_b = F.max_pool2d(x43_1_b, kernel_size=2, stride=2)


        ####################################################

        # siamese processing of input s1_2
        # Stage 1
        x11_b = self.do11_b(F.relu(self.bn11_b(self.conv11_b(s1_2))))
        x12_2_b = self.do12_b(F.relu(self.bn12_b(self.conv12_b(x11_b))))
        x1p_b = F.max_pool2d(x12_2_b, kernel_size=2, stride=2)

        # Stage 2
        x21_b = self.do21_b(F.relu(self.bn21_b(self.conv21_b(x1p_b))))
        x22_2_b = self.do22_b(F.relu(self.bn22_b(self.conv22_b(x21_b))))
        x2p_b = F.max_pool2d(x22_2_b, kernel_size=2, stride=2)

        # Stage 3
        x31_b = self.do31_b(F.relu(self.bn31_b(self.conv31_b(x2p_b))))
        x32_b = self.do32_b(F.relu(self.bn32_b(self.conv32_b(x31_b))))
        x33_2_b = self.do33_b(F.relu(self.bn33_b(self.conv33_b(x32_b))))
        x3p_b = F.max_pool2d(x33_2_b, kernel_size=2, stride=2)

        # Stage 4
        x41_b = self.do41_b(F.relu(self.bn41_b(self.conv41_b(x3p_b))))
        x42_b = self.do42_b(F.relu(self.bn42_b(self.conv42_b(x41_b))))
        x43_2_b = self.do43_b(F.relu(self.bn43_b(self.conv43_b(x42_b))))
        x4p_b = F.max_pool2d(x43_2_b, kernel_size=2, stride=2)

        #################################################### decoder ####################################################
        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = ReplicationPad2d((0, x43_1.size(3) - x4d.size(3), 0, x43_1.size(2) - x4d.size(2)))
        x43_1 = torch.cat((x43_1, x43_1_b), 1)
        x43_2 = torch.cat((x43_2, x43_2_b), 1)

        x4d = torch.cat((pad4(x4d), torch.abs(x43_1 - x43_2)), 1)
        x43d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x42d = self.do42d(F.relu(self.bn42d(self.conv42d(x43d))))
        x41d = self.do41d(F.relu(self.bn41d(self.conv41d(x42d))))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = ReplicationPad2d((0, x33_1.size(3) - x3d.size(3), 0, x33_1.size(2) - x3d.size(2)))
        x33_1 = torch.cat((x33_1, x33_1_b), 1)
        x33_2 = torch.cat((x33_2, x33_2_b), 1)
        x3d = torch.cat((pad3(x3d), torch.abs(x33_1 - x33_2)), 1)
        x33d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x32d = self.do32d(F.relu(self.bn32d(self.conv32d(x33d))))
        x31d = self.do31d(F.relu(self.bn31d(self.conv31d(x32d))))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = ReplicationPad2d((0, x22_1.size(3) - x2d.size(3), 0, x22_1.size(2) - x2d.size(2)))
        x22_1 = torch.cat((x22_1, x22_1_b), 1)
        x22_2 = torch.cat((x22_2, x22_2_b), 1)
        x2d = torch.cat((pad2(x2d), torch.abs(x22_1 - x22_2)), 1)
        x22d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x21d = self.do21d(F.relu(self.bn21d(self.conv21d(x22d))))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = ReplicationPad2d((0, x12_1.size(3) - x1d.size(3), 0, x12_1.size(2) - x1d.size(2)))
        x12_1 = torch.cat((x12_1, x12_1_b), 1)
        x12_2 = torch.cat((x12_2, x12_2_b), 1)
        x1d = torch.cat((pad1(x1d), torch.abs(x12_1 - x12_2)), 1)
        x12d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        x11d = self.conv11d(x12d)

        return x11d
