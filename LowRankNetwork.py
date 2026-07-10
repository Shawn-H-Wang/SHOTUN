import torch
import torch.nn as nn

class CoeffEncoder(nn.Module):
    def __init__(self, in_c, mc):
        super(CoeffEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=mc, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mc),
            nn.ReLU(),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=mc, out_channels=mc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mc),
            nn.ReLU(),
        )
        
        self.cam1 = ChannelAttention(mc, 2)
        self.cam2 = ChannelAttention(mc, 2)

    def forward(self, x):
        fe1 = self.encoder1(x)
        fe1, _ = self.cam1(fe1)
        fe2 = self.encoder2(fe1)
        fe2, ca = self.cam2(fe2)
        return fe2, ca
    

class CoeffDecoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(CoeffDecoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )

        self.cam1 = ChannelAttention(out_c, 2)
        self.cam2 = ChannelAttention(out_c, 2)

    def forward(self, x, ca):
        up_x = self.upsample(x)
        up_x = up_x * ca.expand_as(up_x)
        de_x1 = self.decoder1(up_x)
        de_x1, _ = self.cam1(de_x1)
        de_x2 = self.decoder2(de_x1)
        de_x2, _ = self.cam2(de_x2)
        return de_x2

class ChannelAttention(nn.Module):
    def __init__(self, in_c, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_c, in_c // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_c // reduction_ratio, in_c),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x), y


class CoeffNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(CoeffNet, self).__init__()
        mc = 64
        self.encoder1 = CoeffEncoder(in_c, mc)
        self.encoder2 = CoeffEncoder(mc, mc)
        self.encoder3 = CoeffEncoder(mc, mc)

        self.decoder3 = CoeffDecoder(mc, mc)
        self.decoder2 = CoeffDecoder(mc, mc)
        self.decoder1 = CoeffDecoder(mc, out_c)
        

    def forward(self, x):
        # Encoder 
        ef_x1, ca1 = self.encoder1(x)
        ef_x2, ca2 = self.encoder2(ef_x1)
        ef_x3, ca3 = self.encoder3(ef_x2)

        # Decoder
        df_x3 = self.decoder3(ef_x3, ca3)
        df_x2 = self.decoder2(df_x3, ca2)
        df_x1 = self.decoder1(df_x2, ca1)
        return df_x1



class BasisEncoder(nn.Module):
    def __init__(self, in_c, mc):
        super(BasisEncoder, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_c, mc, 3, 2, 1, bias=False),
            nn.BatchNorm1d(mc),
            nn.Sigmoid()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(mc, mc, 3, 1, 1, bias=False),
            nn.BatchNorm1d(mc),
            nn.Sigmoid()
        )

    def forward(self, x):
        en_x1 = self.encoder1(x)
        en_x2 = self.encoder2(en_x1)
        return en_x2
    

class BasisDecoder(nn.Module):
    def __init__(self, in_c, out_c):
        super(BasisDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)

        self.decoder1 = nn.Sequential(
            nn.Conv1d(in_c*2, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        upx_1 = self.upsample(x1)
        upx_2 = self.upsample(x2)
        de_x = self.decoder1(torch.cat([upx_1, upx_2],dim=1))
        return de_x

class BasisEncoderM(nn.Module):
    def __init__(self, in_c, mc):
        super(BasisEncoderM, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv1d(in_c, mc, 3, 1, 1, bias=False),
            nn.BatchNorm1d(mc),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(mc, mc, 3, 1, 1, bias=False),
            nn.BatchNorm1d(mc),
            nn.ReLU()
        )

    def forward(self, x):
        en_x1 = self.encoder1(x)
        en_x2 = self.encoder2(en_x1)
        return en_x2
    

class BasisDecoderM(nn.Module):
    def __init__(self, in_c, out_c):
        super(BasisDecoderM, self).__init__()

        self.decoder1 = nn.Sequential(
            nn.Conv1d(in_c*2, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        de_x = self.decoder1(torch.cat([x1, x2],dim=1))
        return de_x

class ConvSpectralSubspace(nn.Module):
    def __init__(self, in_c, out_c):
        super(ConvSpectralSubspace, self).__init__()
        mc = 64
        # self.encoder1 = BasisEncoderM(in_c, mc)
        # self.encoder2 = BasisEncoderM(mc, mc)
        # self.encoder3 = BasisEncoderM(mc, mc)

        # self.decoder3 = BasisDecoderM(mc, mc)
        # self.decoder2 = BasisDecoderM(mc, mc)
        self.decoder1 = BasisDecoderM(in_c, out_c)


    def forward(self, x):
        # encoder
        # en_f1 = self.encoder1(x)
        # en_f2 = self.encoder2(en_f1)
        # en_f3 = self.encoder3(en_f2)

        # de_f3 = self.decoder3(en_f3, en_f3)
        # de_f2 = self.decoder2(de_f3, en_f2)
        de_f1 = self.decoder1(x, x)
        return de_f1


class BasisNet(nn.Module):
    def __init__(self, in_c, out_c):
        super(BasisNet, self).__init__()
        mc = 64
        self.encoder1 = BasisEncoderM(in_c, mc)
        self.encoder2 = BasisEncoderM(mc, mc)
        self.encoder3 = BasisEncoderM(mc, mc)

        self.decoder3 = BasisDecoderM(mc, mc)
        self.decoder2 = BasisDecoderM(mc, mc)
        self.decoder1 = BasisDecoderM(mc, out_c)

        self.projection = nn.Sequential(
            nn.Conv1d(out_c, out_c, 1, 1, 0, bias=True),
            # nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.02)
        )


    def forward(self, x):
        # encoder
        en_f1 = self.encoder1(x)
        en_f2 = self.encoder2(en_f1)
        en_f3 = self.encoder3(en_f2)

        de_f3 = self.decoder3(en_f3, en_f3)
        de_f2 = self.decoder2(de_f3, en_f2)
        de_f1 = self.decoder1(de_f2, en_f1)
        return de_f1
        # return self.projection(de_f1)


class BasisHSINet(nn.Module):
    def __init__(self, hsi_c, msi_c, in_c, out_c):
        super(BasisHSINet, self).__init__()
        mc = 64
        self.basis_param = torch.randn([1,out_c,hsi_c]).type(torch.cuda.FloatTensor)
        self.basis_param.data.uniform_(0,1)
        self.basis_param = nn.Parameter(self.basis_param)
        self.out_c = out_c
        self.subspace_dim = in_c
        self.encoder = nn.Sequential(
            nn.Conv2d(hsi_c+msi_c, hsi_c, 3, 1 ,1, bias=False),
            nn.BatchNorm2d(hsi_c),
            nn.Sigmoid()
        )
        self.encoder1 = BasisEncoderM(in_c, mc)
        self.encoder2 = BasisEncoderM(mc, mc)
        self.encoder3 = BasisEncoderM(mc, mc)

        self.decoder3 = BasisDecoderM(mc, mc)
        self.decoder2 = BasisDecoderM(mc, mc)
        self.decoder1 = BasisDecoderM(mc, out_c)


    def forward(self, x):
        """Spectral basis estimited through SVD"""
        # _,_,_Ek=x.reshape(-1, x.shape[-2]*x.shape[-1]).T.svd()
        # _Ek=_Ek.to(device=x.device)
        # _Ek = _Ek[:,:self.out_c].T
        # # _Ek_R = torch.matmul(R, _Ek)
        # # I1 = msi
        # # I2 = hsi
        # # I = torch.matmul(msi.permute(0,1,3,4,2), _Ek_R).permute(0,1,4,2,3)
        # return _Ek
        """DSPN"""
        # x_in = self.encoder(x)
        # en_f0 = torch.cat([nn.functional.avg_pool2d(x_in, x_in.shape[-1]).permute(0,2,3,1).squeeze(0) for _ in range(self.subspace_dim)], dim=1)
        en_f0 = torch.cat([nn.functional.avg_pool2d(x, x.shape[-1]).permute(0,2,3,1).squeeze(0) for _ in range(self.subspace_dim)], dim=1)
        # encoder
        en_f1 = self.encoder1(en_f0)
        en_f2 = self.encoder2(en_f1)
        en_f3 = self.encoder3(en_f2)

        de_f3 = self.decoder3(en_f3, en_f3)
        de_f2 = self.decoder2(de_f3, en_f2)
        de_f1 = self.decoder1(de_f2, en_f1)
        # de_f1 = self.decoder1(en_f1, en_f1)
        return de_f1
        # s = self.basis_param.cuda()
        # return s
    


"""Image as Input: Learnable Deep Low-rank Prior Network"""

# Spectral Basis Learning Network
class BasisLearningNetwork(nn.Module):
    def __init__(self, in_c, out_c):
        super(BasisLearningNetwork, self).__init__()
        mc = 64
        self.encoder1 = BasisEncoderM(in_c, mc)
        self.encoder2 = BasisEncoderM(mc, mc)
        self.encoder3 = BasisEncoderM(mc, mc)

        self.decoder3 = BasisDecoderM(mc, mc)
        self.decoder2 = BasisDecoderM(mc, mc)
        self.decoder1 = BasisDecoderM(mc, out_c)

    def forward(self, x):
        # encoder
        en_f1 = self.encoder1(x)
        en_f2 = self.encoder2(en_f1)
        en_f3 = self.encoder3(en_f2)

        de_f3 = self.decoder3(en_f3, en_f3)
        de_f2 = self.decoder2(de_f3, en_f2)
        de_f1 = self.decoder1(de_f2, en_f1)
        return de_f1
    


class FMSAN(nn.Module):

    def __init__(self, channel):
        super(FMSAN, self).__init__()
        
        mid_channel = 32

        self.linear = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.LeakyReLU(0.02),
            nn.Linear(in_features=mid_channel, out_features=mid_channel),
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        a = self.linear(x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]*x.shape[4])).squeeze(0)
        a = self.softmax(a@a.T)
        a = a - torch.diag_embed(torch.diag(a))
        return a