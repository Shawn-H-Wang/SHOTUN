# BlindTest estimation network
# Author: JianJun Liu
# Date: 2022-1-13
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as fun
from utils import toolkits, torchkits, DataInfo, BlurDown, BlurDownOri, BlurUp

class BlindNet2(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio
        self.pad_num = int((self.ker_size - 1) / 2)
        psf_1 = torch.ones([1, 1, self.ker_size, 1]) * (1.0 / (self.ker_size)) # torch.ones([1, 1, self.ker_size, 1]) * 5e-1
        psf_2 = torch.ones([1, 1, 1, self.ker_size]) * (1.0 / (self.ker_size)) # torch.ones([1, 1, 1, self.ker_size]) * 5e-1

        # psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        # self.psf = nn.Parameter(psf)

        # psf_t1 = torch.ones([1, 1, self.ker_size, 1]) * 1e-4 # torch.ones([1, 1, self.ker_size, 1]) * 5e-1
        # psf_t2 = torch.ones([1, 1, 1, self.ker_size]) * 1e-4 # torch.ones([1, 1, 1, self.ker_size]) * 5e-1
        # psf_1 = torch.nn.init.kaiming_normal_(psf_1)
        # psf_2 = torch.nn.init.kaiming_normal_(psf_2)
        self.psf_1 = nn.Parameter(psf_1)
        self.psf_2 = nn.Parameter(psf_2)
        # self.psf_t1 = nn.Parameter(psf_t1)
        # self.psf_t2 = nn.Parameter(psf_t2)
        # self.srf = torch.from_numpy(sio.loadmat("D:\\Dataset\\HySure-master\\demos\\R.mat")["R_est"]).float().unsqueeze(2).unsqueeze(2).cuda()
        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands)
        # srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1])
        # srf = torch.nn.init.trunc_normal_(srf, std=0.1)
        self.srf = nn.Parameter(srf)
        self.blur_down = BlurDown()
        self.my_upsample = BlurUp()

    def forward(self, X):
        srf_div = torch.sum(self.srf, dim=1, keepdim=True)
        srf_div = torch.div(1.0, srf_div)
        srf_div = torch.transpose(srf_div, 0, 1)  # 1 x l x 1 x 1
        Ylow = fun.conv2d(X, self.srf, None)
        Ylow = torch.mul(Ylow, srf_div)
        Ylow = torch.clamp(Ylow, 0.0, 1.0)
        Zlow = self.blur_down(X, self.psf_1, self.psf_2, self.pad_num, self.ms_bands, self.ratio)
        Zlow = torch.clamp(Zlow, 0.0, 1.0)
        # Y_up = self.my_upsample(Ylow, self.psf_t1, self.psf_t2, self.pad_num, self.ms_bands, self.ratio)
        # Z_up = torch.div(Zlow, srf_div)
        # Z_up = fun.conv2d(Z_up, self.srf.permute(1,0,2,3), None)
        # Y_up = torch.clamp(Y_up, 0.0, 1.0)
        # Z_up = torch.clamp(Z_up, 0.0, 1.0)
        return Ylow, Zlow # , Y_up, Z_up


class BlindNet(nn.Module):
    def __init__(self, hs_bands, ms_bands, ker_size, ratio):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size
        self.ratio = ratio
        self.pad_num = int((self.ker_size - 1) / 2)
        psf_1 = torch.ones([1, 1, self.ker_size, 1]) * (1.0 / (self.ker_size)) # torch.ones([1, 1, self.ker_size, 1]) * 5e-1
        psf_2 = torch.ones([1, 1, 1, self.ker_size]) * (1.0 / (self.ker_size)) # torch.ones([1, 1, 1, self.ker_size]) * 5e-1

        # psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        # self.psf = nn.Parameter(psf)

        # psf_t1 = torch.ones([1, 1, self.ker_size, 1]) * 1e-4 # torch.ones([1, 1, self.ker_size, 1]) * 5e-1
        # psf_t2 = torch.ones([1, 1, 1, self.ker_size]) * 1e-4 # torch.ones([1, 1, 1, self.ker_size]) * 5e-1
        # psf_1 = torch.nn.init.kaiming_normal_(psf_1)
        # psf_2 = torch.nn.init.kaiming_normal_(psf_2)
        self.psf_1 = nn.Parameter(psf_1)
        self.psf_2 = nn.Parameter(psf_2)
        # self.psf_t1 = nn.Parameter(psf_t1)
        # self.psf_t2 = nn.Parameter(psf_t2)
        # self.srf = torch.from_numpy(sio.loadmat("D:\\Dataset\\HySure-master\\demos\\R.mat")["R_est"]).float().unsqueeze(2).unsqueeze(2).cuda()
        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands)
        # srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1])
        # srf = torch.nn.init.trunc_normal_(srf, std=0.1)
        self.srf = nn.Parameter(srf)
        self.blur_down = BlurDown(stride=0)
        self.my_upsample = BlurUp()

    def forward(self, Y, Z):
        srf_div = torch.sum(self.srf, dim=1, keepdim=True)
        srf_div = torch.div(1.0, srf_div)
        srf_div = torch.transpose(srf_div, 0, 1)  # 1 x l x 1 x 1
        Ylow = fun.conv2d(Y, self.srf, None)
        Ylow = torch.mul(Ylow, srf_div)
        Ylow = torch.clamp(Ylow, 0.0, 1.0)
        Zlow = self.blur_down(Z, self.psf_1, self.psf_2, self.pad_num, self.ms_bands, self.ratio)
        Zlow = torch.clamp(Zlow, 0.0, 1.0)
        # Y_up = self.my_upsample(Ylow, self.psf_t1, self.psf_t2, self.pad_num, self.ms_bands, self.ratio)
        # Z_up = torch.div(Zlow, srf_div)
        # Z_up = fun.conv2d(Z_up, self.srf.permute(1,0,2,3), None)
        # Y_up = torch.clamp(Y_up, 0.0, 1.0)
        # Z_up = torch.clamp(Z_up, 0.0, 1.0)
        return Ylow, Zlow # , Y_up, Z_up


class Blind(DataInfo):
    def __init__(self, ndata, nratio, nsnr=0, data_iter=1, blind=True):
        super().__init__(ndata, nratio, nsnr, data_iter)
        self.strBR = 'BR.mat'
        self.blind = blind
        if self.blind is False:
            # self.psf, self.srf
            print('using true psf and srf!')
            return
        print('estimate psf and srf ...')
        # set
        self.lr = 5e-5  # learning rate
        self.ker_size = 9 # 2 * self.ratio - 1  # spatial blur kernel size
        # variable, graph and etc.
        self.__hsi = torch.tensor(self.hsi)
        self.__msi = torch.tensor(self.msi)
        self.model = BlindNet(self.hs_bands, self.ms_bands, self.ratio, self.ratio).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        toolkits.check_dir(self.model_save_path)

    def train(self, max_iter=50000, verb=True):
        if self.blind is False:
            return
        hsi, msi = self.__hsi.cuda(), self.__msi.cuda()
        loss_epoch = np.inf
        for epoch in range(0, max_iter):
            Ylow, Zlow = self.model(hsi, msi)
            # loss_1 = torchkits.torch_norm(Ylow - Zlow)
            loss_1 = torchkits.torch_ssim_loss(Ylow, Zlow)
            # loss_2 = torchkits.torch_norm(Y_up - msi)
            # loss_3 = torchkits.torch_norm(Z_up - hsi)
            loss = loss_1 # + 0 * (loss_2 + loss_3)
            if verb is True:
                if (epoch + 1) % 100 == 0:
                    if loss > loss_epoch:
                        break
                    else:
                        loss_epoch = loss
                    print('epoch: %s, lr: %s, loss: %s' % (epoch + 1, self.lr, loss))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.apply(self.check_weight)
        torch.save(self.model.state_dict(), self.model_save_path + 'parameter.pkl')
        self.psf_1 = torch.tensor(torchkits.to_numpy(self.model.psf_1.data))
        self.psf_2 = torch.tensor(torchkits.to_numpy(self.model.psf_2.data))
        # self.psf = torch.tensor(torchkits.to_numpy(self.model.psf.data))
        # self.psf_t1 = torch.tensor(torchkits.to_numpy(self.model.psf_t1.data))
        # self.psf_t2 = torch.tensor(torchkits.to_numpy(self.model.psf_t2.data))
        self.srf = torch.tensor(torchkits.to_numpy(self.model.srf.data))

    def get_save_result(self, is_save=True):
        if self.blind is False:
            return
        print('save psf and srf ...')
        self.model.load_state_dict(torch.load(self.model_save_path + 'parameter.pkl'))
        # mat = sio.loadmat(self.save_path + self.strBR)
        self.psf_1 = torch.tensor(torchkits.to_numpy(self.model.psf_1.data))
        self.psf_2 = torch.tensor(torchkits.to_numpy(self.model.psf_2.data))
        # self.psf = torch.tensor(torchkits.to_numpy(self.model.psf.data))
        # self.psf_t1 = torch.tensor(torchkits.to_numpy(self.model.psf_t1.data))
        # self.psf_t2 = torch.tensor(torchkits.to_numpy(self.model.psf_t2.data))
        self.srf = torch.tensor(torchkits.to_numpy(self.model.srf.data))
        # self.srf = torch.from_numpy(mat["R"]).float()
        if is_save is True:
            psf1 = torchkits.to_numpy(self.model.psf_1.data)
            psf2 = torchkits.to_numpy(self.model.psf_2.data)
            # psf = torchkits.to_numpy(self.model.psf.data)
            # psf_t1 = torchkits.to_numpy(self.model.psf_t1.data)
            # psf_t2 = torchkits.to_numpy(self.model.psf_t2.data)
            srf = torchkits.to_numpy(self.model.srf.data)
            # psf1 = np.squeeze(psf1)
            # psf2 = np.squeeze(psf2)
            # psf = np.squeeze(psf)
            # psf_t1 = np.squeeze(psf_t1)
            # psf_t2 = np.squeeze(psf_t2)
            srf = np.squeeze(srf)  # b x B
            self.psf_1, self.psf_2, self.srf = psf1, psf2, srf
            sio.savemat(self.save_path + self.strBR, {'B1': psf1,'B2': psf2, 'R': srf})
        return

    @staticmethod
    def check_weight(model):
        if hasattr(model, 'psf'):
            w = model.psf.data
            w.clamp_(0.0, 1.0)
            psf_div = torch.sum(w)
            psf_div = torch.div(1.0, psf_div)
            w.mul_(psf_div)
        if hasattr(model, 'srf'):
            w = model.srf.data
            w.clamp_(0.0, 10.0)
            srf_div = torch.sum(w, dim=1, keepdim=True)
            srf_div = torch.div(1.0, srf_div)
            w.mul_(srf_div)
