import numpy as np
import scipy.io as sio
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as fun
import torch.utils.data as data
from pysptools.material_count import HySime
from skimage.restoration import  denoise_nl_means,estimate_sigma

from utils_blocks import block_module
from utils import toolkits, torchkits, DataInfo, BlurDown, PatchDataset, CubeDataset
from blind import Blind, BlindNet
from LowRankNetwork import BasisNet, CoeffNet, ConvSpectralSubspace, BasisHSINet
from metrics import MetricsCal

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def soft_threshold(x, lambd):
    return nn.functional.relu(x - lambd,inplace=True) - nn.functional.relu(-x - lambd,inplace=True)

def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def mode_product(tensor, factor_matrix, mode):
    tensor_dims = tensor.size()
    reshaped_tensor = torch.moveaxis(tensor, mode, -1)

    mode_size = tensor_dims[mode]
    reshaped_tensor = reshaped_tensor.reshape(-1, mode_size)

    result = torch.matmul(reshaped_tensor, factor_matrix)

    result_dims = list(tensor_dims)
    result_dims[mode] = factor_matrix.size(1)
    result = result.view(result_dims)

    return result

def chain_mode_product(tensor, factor_matrices):
    for i in range(1,len(factor_matrices)+1):
        tensor = mode_product(tensor, factor_matrices[i-1], i)
    return tensor

def chain_mode_product_T(tensor, factor_matrices):
    for i in range(1,len(factor_matrices)+1):
        tensor = mode_product(tensor, factor_matrices[i-1].t(), i)
    return tensor


class HOTN(nn.Module):
    def __init__(self, core_size, cube_size, threshold, label):
        super(HOTN, self).__init__()
        self.label = label
        D=[]
        D.append(torch.Tensor(core_size[0], cube_size[0]).float()) # torch.eye(core_size[3]).float()
        D.append(torch.Tensor(core_size[1], cube_size[1]).float()) # torch.eye(core_size[3]).float()
        D.append(torch.Tensor(core_size[2], cube_size[2]).float()) # torch.eye(core_size[3]).float()
        D.append(torch.Tensor(core_size[3], cube_size[3]).float()) # torch.eye(core_size[3]).float()
        B=[]
        
        self.apply_D=nn.ParameterList()
        for i in range(4):
            stdv = 1 / math.sqrt(D[i].size(0))
            D[i].data.uniform_(-stdv, stdv)
            B.append(torch.clone(D[i]))
            self.apply_D.append(nn.Parameter(B[-1]))
        
        self.lmbda = nn.Parameter(torch.zeros([1,core_size[3],core_size[2],core_size[0],core_size[1]]))
        nn.init.constant_(self.lmbda, threshold)

    def forward(self, ct):
        """Self-Expressive"""
        # """Auto-Regressive"""
        D4 = self.apply_D[3].tril(diagonal=0).t() + self.apply_D[3].tril(diagonal=-1)
        D4 = (D4.t() @ D4).clamp_(0, D4.max().item())
        D4 = (D4 - torch.diag_embed(torch.diag(D4))).clamp_(0,1)
        """Decode"""
        clean_spatial_cubes = chain_mode_product(ct, [D4, self.apply_D[2], self.apply_D[0], self.apply_D[1]])
        clean_spatial_cubes_3D = chain_mode_product(ct, [torch.eye(D4.shape[0]).cuda(), self.apply_D[2], self.apply_D[0], self.apply_D[1]])
        return clean_spatial_cubes, clean_spatial_cubes_3D, D4


class CoreTensorUpdateStage(nn.Module):
    def __init__(self, cube_size, core_size, multi_lmbda, unfoldings, threshold):
        super(CoreTensorUpdateStage, self).__init__()
        self.core_size = core_size
        self.cube_size = cube_size
        
        self.soft_threshold = soft_threshold
        
        """Initialize the high order tensor"""
        self.center = torch.Tensor(1,core_size[3],core_size[2],core_size[0],core_size[1]).type(torch.cuda.FloatTensor)
        self.center.data.uniform_(0,1)
        # self.center = nn.Parameter(self.center)
    
    def forward(self, msi, hsi, _Ek, _EkT, apply_D, srf, p1, p2, lmbda):
        R = srf.squeeze().cuda()
        conv = (p1@p2.permute(0,2,1)).squeeze()
        I1 = msi
        I2 = hsi
        thresh_fn = self.soft_threshold
        apply_D = apply_D.cuda()
        PD1 = self.center.cuda()
        gamma_k = thresh_fn(PD1, 0.001)
        mean_patch = gamma_k.mean(dim=1, keepdim=True)
        gamma_k = gamma_k - mean_patch
        gamma_k = thresh_fn(gamma_k, 0.001)
        """Self-Expressive"""
        D4 = apply_D[3].tril(diagonal=0).t() + apply_D[3].tril(diagonal=-1)
        D4 = (D4.t() @ D4).clamp_(0,D4.max().item())
        D4 = (D4 - torch.diag_embed(torch.diag(D4))).clamp_(0,1)
        """Update stage"""
        up_rec = chain_mode_product(gamma_k, [D4, apply_D[2]@_Ek.t(), apply_D[0], apply_D[1]]).clamp_(0,1)
        x_k = (up_rec.permute(0,1,3,4,2)@R.t()).permute(0,1,4,2,3)
        y_k = fun.conv2d(up_rec.squeeze(), p1.repeat(up_rec.shape[2],1,1,1), None, (nratio,1), (int((nratio - 1) / 2),0), groups=up_rec.shape[2])
        y_k = fun.conv2d(y_k, p2.permute(0,2,1).repeat(up_rec.shape[2],1,1,1), None, (1,nratio), (0,int((nratio - 1) / 2)), groups=up_rec.shape[2]).unsqueeze(0)
        res1 = x_k - I1
        res2 = y_k - I2
        r_k1 = chain_mode_product(res1, [D4.t(), R@_EkT.t()@apply_D[2].t(), apply_D[0].t(), apply_D[1].t()])
        """Circular Matrix Computation for Gaussian Filter"""
        res2_up = res2.reshape(res2.shape[1],res2.shape[2],self.cube_size[0]//nratio,self.cube_size[1]//nratio,1,1).repeat(1,1,1,1,nratio,nratio) \
            * conv.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.core_size[-1], hsi.shape[2], hsi.shape[3], hsi.shape[4], 1, 1)
        res2_up = res2_up.permute(0,1,2,4,3,5).reshape(-1,hsi.shape[2],self.cube_size[0],self.cube_size[1]).unsqueeze(0).clamp_(0,1)
        r_k2 = chain_mode_product(res2_up, [D4.t(), _EkT.t()@apply_D[2].t(), apply_D[0].t(), apply_D[1].t()])
        r_k = r_k1 + r_k2
        lmbda_ = lmbda
        gamma_k = thresh_fn(gamma_k - r_k, lmbda_)
        return gamma_k


class DSPN(nn.Module):
    def __init__(self, hs_bands, ms_bands, cube_size):
        super(DSPN, self).__init__()
        self.basis_network_hsi_input = BasisHSINet(hsi_c=hs_bands, msi_c=ms_bands, in_c=cube_size[2], out_c=cube_size[2]).cuda()
        
    def forward(self, z, y, p1, p2):
        basis = self.basis_network_hsi_input(z)
        _Ek = basis.squeeze(0).t()
        return _Ek


class AggerationModule(nn.Module):
    def __init__(self, original_indices):
        super(AggerationModule, self).__init__()
        self.original_indices = original_indices

    def forward(self, clean_spatial_cubes, S, block):
        clean_spatial = block._agregate_blocks(clean_spatial_cubes.squeeze(0))
        x_out = torch.matmul(clean_spatial.permute(0,2,3,1), S.t()).permute(0,3,1,2).clamp_(0,1)
        return x_out
    

class UNFOLDNET(DataInfo):

    def __init__(self, ndata, nratio, nsnr=0, data_iter=1, psf=None, srf=None, edm_num=80, stage=3):
        super().__init__(ndata, nratio, nsnr, data_iter)
        self.strX = self.data_names[0] # 'ARAD_1K_0911.mat'
        if psf is not None:
            self.psf = psf
        if srf is not None:
            self.srf = srf

        # set
        self.lr = 5e-3  # learning rate
        self.edm_num = edm_num  # (like) endmember number
        self.ker_size = self.psf[1].shape[-1]  # spatial blur kernel size
        self.lam_A, self.lam_B, self.lam_C = 1, 1, 1e-3  # weights for spectral term, spatial term and weight decay
        self.lr_fun = lambda epoch: (1.0 - max(0, epoch + 1 - 800) / 1200)  # decay of learning rate

        # define
        self.psf_1 = np.reshape(self.psf[0], newshape=(1, 1, self.ker_size, 1))
        self.psf_1 = torch.tensor(self.psf_1).float().cuda()
        self.psf_1_hs = self.psf_1.repeat(self.hs_bands, 1, 1, 1)
        self.psf_2 = np.reshape(self.psf[1], newshape=(1, 1, 1, self.ker_size))
        self.psf_2 = torch.tensor(self.psf_2).float().cuda()
        self.psf_2_hs = self.psf_2.repeat(self.hs_bands, 1, 1, 1)
        self.srf = np.reshape(self.srf, newshape=(self.ms_bands, self.hs_bands, 1, 1))
        self.srf = torch.tensor(self.srf).float().cuda()

        # variable, graph and etc
        self.__hsi = torch.tensor(self.hsi)
        _, r_h, r_w, c = self.hsi.shape
        self.SNR=35
        add_noise = 0
        sigmah = add_noise * np.sqrt(np.sum(self.hsi)**2 / (10 ** (self.SNR/10)) / (r_h*r_w*c))
        self.hsi_noised = torch.tensor(self.hsi + sigmah * np.random.normal(size=np.shape(self.hsi))).float()
        self.__msi = torch.tensor(self.msi)
        _, r_h, r_w, c = self.msi.shape
        sigmam = add_noise * np.sqrt(np.sum(self.msi)**2 / (10 ** (self.SNR/10)) / (r_h*r_w*c))
        self.msi_noised = torch.tensor(self.msi + sigmam * np.random.normal(size=np.shape(self.msi))).float()
        self.ref_tensor = torch.tensor(self.ref).float().unsqueeze(0).permute(0,3,1,2)
        self.hsi_up = nn.Upsample(scale_factor=self.ratio, mode='bicubic', align_corners=False)(self.hsi_noised)
        params = {
            'crop_out_blocks': 0,
            'ponderate_out_blocks': 1,
            'sum_blocks': 0,
            'pad_even': 1,  # otherwise pad with 0 for las
            'centered_pad': 0,  # corner pixel have only one estimate
            'pad_block': False,  # pad so each pixel has S**2 estimate
            'pad_patch': False,  # pad so each pixel from the image has at le
            'no_pad': True,
            'custom_pad': False,
            'avg': 1
        }
        # Parameter of cube and core-tensor size
        cs = 16 # cube size
        cr = 16 # core size
        ws = 8 # step size
        ss = 4 # spectral size
        self.core_size = [cr,cr,ss,-1]
        self.cube_size = [cs,cs,ss,-1]
        cube_size_hsi = [cs//nratio,cs//nratio,ss,-1]
        multi_lmbda = False
        unfoldings = 1
        self.K = unfoldings
        threshold = 0.001
        self.block = block_module(self.cube_size[0], ws, 9, params)
        self.block_hsi = block_module(cube_size_hsi[0], ws//nratio, 9, params)
        self.msi_cubes = self.block._make_blocks(self.msi_noised)
        self.ref_cubes = self.block._make_blocks(self.ref_tensor)
        self.hsi_cubes = self.block_hsi._make_blocks(self.hsi_noised)
        self.hsi_up_cubes = self.block._make_blocks(self.hsi_up)
        self.cube_size[-1] = self.msi_cubes.shape[0]
        self.core_size[-1] = self.msi_cubes.shape[0] 

        # sort the spilt cubes
        from sklearn.decomposition import SparseCoder
        from sklearn.cluster import SpectralClustering, KMeans
        from sklearn.neighbors import kneighbors_graph
        patches_flat = self.msi_cubes.detach().cpu().numpy().reshape([self.cube_size[-1],self.cube_size[0]*self.cube_size[1]*self.msi_noised.shape[1]])
        num_patches = patches_flat.shape[0]
        dictionary = patches_flat  # 使用patch本身作为字典
        coder = SparseCoder(dictionary=dictionary, transform_algorithm='lasso_lars', transform_alpha=0.1)
        sparse_coefficients = coder.transform(patches_flat)
        similarity_matrix = np.abs(sparse_coefficients) + np.abs(sparse_coefficients.T)
        num_clusters = 6  # 假设聚类数为10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(patches_flat)
        self.labels = kmeans.labels_
        self.labels = torch.from_numpy(self.labels)
        self.original_indices = torch.arange(len(self.msi_cubes))
        self.sorted_labels, self.sorted_indices = torch.sort(self.labels)

        self.l2loss = nn.MSELoss(size_average=False).cuda()

        self.dspn = DSPN(hs_bands=self.hs_bands, ms_bands=self.ms_bands, cube_size=self.cube_size).cuda()
        self.optimizer_dspn = optim.Adam(self.dspn.parameters(), lr=self.lr, weight_decay=self.lam_C)
        self.ctus = CoreTensorUpdateStage(self.cube_size, self.core_size, multi_lmbda, unfoldings, threshold).cuda()
        self.hotn = HOTN(core_size=self.core_size, cube_size=self.cube_size, threshold=threshold, label=self.labels).cuda()
        self.optimizer_hotn = optim.Adam(self.hotn.parameters(), lr=self.lr, weight_decay=self.lam_C)
        self.aggerate = AggerationModule(self.original_indices)

        self.blindnet = BlindNet(self.hs_bands, self.ms_bands, self.ratio, self.ratio).cuda()
        self.blindnet.load_state_dict(torch.load('./data/pavia/Blind{}0/model/parameter.pkl'.format(self.ratio)))
        self.optimizer_blindnet = optim.Adam(self.blindnet.parameters(), lr=5e-5)
        self.scheduler_dspn = optim.lr_scheduler.LambdaLR(self.optimizer_dspn, self.lr_fun)
        self.scheduler_hotn = optim.lr_scheduler.LambdaLR(self.optimizer_hotn, self.lr_fun)
        toolkits.check_dir(self.model_save_path)
        torchkits.get_param_num(self.dspn)
        torchkits.get_param_num(self.hotn)
        self.hs_border = math.ceil((self.ker_size - 1) / 2 / self.ratio)  # remove the pixels effected by spatial blur
        self.ms_border = self.hs_border * self.ratio  # remove the pixels effected by spatial blur
        self.dataset = CubeDataset(self.msi_cubes.unsqueeze(0), self.__hsi.unsqueeze(0))
        self.blur_down = BlurDown(stride=nratio)
        pass

    def cpt_target(self, X):
        Z = self.blur_down(X, self.psf_1_hs, self.psf_2_hs, int((self.ker_size - 1) / 2), self.hs_bands, self.ratio)
        Y = fun.conv2d(X, self.srf, None)
        return Y, Z
      
    def build_loss(self, Y, Z, hsi, msi): # , basis, coeff, rec_z
        rec_loss = self.l2loss(Y, hsi) + self.l2loss(Z, msi)
        loss = rec_loss
        return loss

    def train(self, max_iter=2000, verb=True, is_save=True):
        # train ...
        loader = data.DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
        iteration, epoch = 0, 0
        # self.model.train()
        self.dspn.train()
        self.ctus.train()
        self.hotn.train()
        X_iter = self.hsi_up.cuda()
        Z_ref = Variable(self.__hsi, requires_grad=True).cuda()
        Y_ref = Variable(self.__msi, requires_grad=True).cuda()
        Z_noised = Variable(self.hsi_noised, requires_grad=True).cuda()
        Y_noised = Variable(self.msi_noised, requires_grad=True).cuda()
        msi_cube = Variable(self.msi_cubes.unsqueeze(0), requires_grad=True).cuda()
        hsi_cube = Variable(self.hsi_cubes.unsqueeze(0), requires_grad=True).cuda()
        hsi_up_cube = Variable(self.hsi_up_cubes.unsqueeze(0), requires_grad=True).cuda()
        metric_data = [["Epoch","Training Loss","RMSE","PSNR","SAM","ERGAS","SSIM","UIQI"]]
        phi = 5e-3
        time_start = time.perf_counter()
        while True:
            num1_epochs = 1
            num2_epochs = 7
            for num1 in range(num1_epochs):
                S = self.dspn(Z_noised, Y_noised, self.psf_1[0], self.psf_2[0].permute(0,2,1))
                for num2 in range(num2_epochs):
                    S = self.dspn.basis_network_hsi_input(Z_noised).squeeze(0).t()
                    ST = S.t()
                    ct = self.ctus(msi_cube, hsi_cube, S, ST, self.hotn.apply_D, self.blindnet.srf, self.blindnet.psf_1[0], self.blindnet.psf_2[0].permute(0,2,1), self.hotn.lmbda)
                    A, A_3D, D4 = self.hotn(ct)
                    X_out = self.aggerate(A, S, self.block)
                    X_out_3D = self.aggerate(A_3D, S, self.block)
                    Yhat, Zhat = self.cpt_target(X_out)
                    Yhat_3D, Zhat_3D = self.cpt_target(X_out_3D)
                    loss = self.build_loss(Zhat, Yhat, Z_ref, Y_ref) + + self.build_loss(Zhat_3D, Yhat_3D, Z_ref, Y_ref)
                    if num2+1 == num2_epochs:
                        self.optimizer_dspn.zero_grad()
                        self.optimizer_hotn.zero_grad()
                        self.optimizer_blindnet.zero_grad()
                        loss.backward()
                        self.optimizer_dspn.step()
                        self.optimizer_hotn.step()
                        self.optimizer_blindnet.step()
                    else:
                        self.optimizer_hotn.zero_grad()
                        loss.backward()
                        self.optimizer_hotn.step()
            iteration += 1
            self.scheduler_dspn.step()
            self.scheduler_hotn.step()
 
            if iteration % 20 == 0:
                self.dspn.eval()
                self.ctus.eval()
                self.hotn.eval()
                lr = self.optimizer_hotn.param_groups[0]['lr']
                Xh = torchkits.to_numpy(X_out)
                Xh = toolkits.channel_last(Xh)
                Ds = self.hotn.apply_D

                """simulated fusion experiments"""
                rmse, psnr, sam, ergas, ssim, uiqi = MetricsCal(self.ref, Xh, scale=8)
                message = 'iter/epoch: %s, lr: %s, rmse: %s, psnr: %s, sam: %s, ergas: %s, ssim: %s, uiqi: %s, loss: %s' % (iteration, 
                                                                                   lr, 
                                                                                   round(rmse.item(), 4),
                                                                                   round(psnr.item(),6), 
                                                                                   round(sam.item(),6), 
                                                                                   round(ergas.item(), 2),
                                                                                   round(ssim.item(), 3),
                                                                                   round(uiqi.item(), 3),
                                                                                   loss.item())
                print(message)
                with open(self.save_path + "precision_full_unfolded.txt", mode="a+") as f:
                    f.write(message+"\n")
                # self.model.train()
                self.dspn.train()
                self.ctus.train()
                self.hotn.train()

            if iteration >= max_iter:
                break

        time_end = time.perf_counter()
        train_time = time_end - time_start
        print('running time %ss' % train_time)

        """Save results to excel tabel"""
        import xlwt
        metric_data = np.array(metric_data)
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)

        for i in range(metric_data.shape[0]):
            for j in range(metric_data.shape[1]):
                sheet.write(i, j, metric_data[i][j])

        pass

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    setup_seed(5) # 5 for Pavia
    ndata, nratio, nsnr = 0, 8, 0
    stage, edm_num = 3, 40  # pavia: 3, 80; ksc: 3, 80; dc: 3, 30; UH: 3, 30
    blind = Blind(ndata=ndata, nratio=nratio, nsnr=nsnr, blind=True)
    blind.train()
    
    net = UNFOLDNET(ndata=ndata, nratio=nratio, nsnr=nsnr, psf=[blind.psf_1, blind.psf_2], srf=blind.srf, stage=stage, edm_num=edm_num)
    net.train(verb=True, is_save=True)


# class TISTANet(nn.Module):
#     def __init__(self, ms_bands, hs_bands, core_size, cube_size, multi_lmbda, unfoldings, threshold):
#         super(TISTANet, self).__init__()
#         self.multi_lmbda = multi_lmbda
#         self.unfoldings = unfoldings
#         self.cube_size = cube_size
#         self.core_size = core_size

#         D=[]
#         D.append(torch.Tensor(core_size[0],cube_size[0]).float()) # torch.eye(core_size[3]).float()
#         D.append(torch.Tensor(core_size[1],cube_size[1]).float()) # torch.eye(core_size[3]).float()
#         D.append(torch.Tensor(core_size[2],cube_size[2]).float()) # torch.eye(core_size[3]).float()
#         D.append(torch.Tensor(core_size[3],cube_size[3]).float()) # torch.eye(core_size[3]).float()
#         B=[]
        
#         self.apply_D=nn.ParameterList()
#         for i in range(4):
#             if i<4:
#                 stdv = 1 / math.sqrt(D[i].size(0))
#                 D[i].data.uniform_(-stdv, stdv)
#             B.append(torch.clone(D[i]))
#             self.apply_D.append(nn.Parameter(B[-1]))
#         if multi_lmbda:
#             self.lmbda = nn.ParameterList(
#             [nn.Parameter(torch.zeros([1,core_size[3],core_size[2],core_size[0],core_size[1]])) for _ in range(unfoldings)])
#             [nn.init.constant_(x, threshold) for x in self.lmbda]
#         else:
#             self.lmbda = nn.Parameter(torch.zeros([1,core_size[3],core_size[2],core_size[0],core_size[1]]))
#             nn.init.constant_(self.lmbda, threshold)

#         self.center = torch.Tensor(1,core_size[3],core_size[2],core_size[0],core_size[1]).type(torch.cuda.FloatTensor)
#         self.center.data.uniform_(0,1)
#         self.center = nn.Parameter(self.center)

#         self.soft_threshold = soft_threshold

#         # Deep Low-rankness Network
#         self.basis_network_hsi_input = BasisHSINet(hsi_c=hs_bands, msi_c=ms_bands, in_c=cube_size[2], out_c=cube_size[2]).cuda()
        

#     def forward(self, z, y, msi, hsi, srf, p1, p2, block:block_module):
#         R = srf.squeeze().cuda()
#         conv = (p1@p2.permute(0,2,1)).squeeze()
    
#         """Deep Spectral Prior Network"""
#         y_down = fun.conv2d(y, conv.repeat(y.shape[1],1,1,1), None, (nratio,nratio), (int((nratio - 1) / 2),int((nratio - 1) / 2)), groups=y.shape[1])
#         basis = self.basis_network_hsi_input(torch.cat([z, y_down], dim=1))
#         _Ek = basis.squeeze(0).t()
#         I1 = msi
#         I2 = hsi
        
#         thresh_fn = self.soft_threshold
#         self.apply_D = self.apply_D.cuda()
#         PD1 = self.center.cuda()
#         gamma_k = thresh_fn(PD1, 0.001)
#         mean_patch = gamma_k.mean(dim=1, keepdim=True)
#         gamma_k = gamma_k - mean_patch
#         gamma_k = thresh_fn(gamma_k, 0.001)

#         """Self-Expressive"""
#         D4 = self.apply_D[3].tril(diagonal=0).t() + self.apply_D[3].tril(diagonal=-1)
#         D4 = (D4.t() @ D4).clamp_(0,D4.max().item())
#         D4 = (D4 - torch.diag_embed(torch.diag(D4))).clamp_(0,1)
#         # D4_l = []
#         # for idx in range(3,len(self.apply_D)):
#         #     D4 = self.apply_D[idx].tril(diagonal=0).t() + self.apply_D[idx].tril(diagonal=-1)
#         #     D4 = (D4.t() @ D4).clamp_(0, D4.max().item())
#         #     D4 = (D4 - torch.diag_embed(torch.diag(D4))).clamp_(0,1)
#         #     D4_l.append(D4)
#         # D4 = torch.block_diag(*D4_l)
#         # D3 = torch.eye(self.apply_D[2].shape[-1]).cuda()
#         for k in range(self.unfoldings):
#             up_rec = chain_mode_product(gamma_k, [D4, self.apply_D[2]@_Ek.t(), self.apply_D[0], self.apply_D[1]]).clamp_(0,1)
#             # up_rec = chain_mode_product(gamma_k, [D4, D3@_Ek.t(), self.apply_D[0], self.apply_D[1]]).clamp_(0,1)
#             x_k = (up_rec.permute(0,1,3,4,2)@R.t()).permute(0,1,4,2,3)
#             y_k = fun.conv2d(up_rec.squeeze(), p1.repeat(up_rec.shape[2],1,1,1), None, (nratio,1), (int((nratio - 1) / 2),0), groups=up_rec.shape[2])
#             y_k = fun.conv2d(y_k, p2.permute(0,2,1).repeat(up_rec.shape[2],1,1,1), None, (1,nratio), (0,int((nratio - 1) / 2)), groups=up_rec.shape[2]).unsqueeze(0)
#             res1 = x_k - I1
#             res2 = y_k - I2
#             r_k1 = chain_mode_product(res1, [D4.t(), R@_Ek@self.apply_D[2].t(), self.apply_D[0].t(), self.apply_D[1].t()])
#             # r_k1 = chain_mode_product(res1, [D4.t(), R@_Ek@D3.t(), self.apply_D[0].t(), self.apply_D[1].t()])
#             """Circular Matrix Computation for Gaussian Filter"""
#             res2_up = res2.reshape(res2.shape[1],res2.shape[2],self.cube_size[0]//nratio,self.cube_size[1]//nratio,1,1).repeat(1,1,1,1,nratio,nratio) \
#                 * conv.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(self.core_size[-1], hsi.shape[2], hsi.shape[3], hsi.shape[4], 1, 1)
#             res2_up = res2_up.permute(0,1,2,4,3,5).reshape(-1,hsi.shape[2],self.cube_size[0],self.cube_size[1]).unsqueeze(0).clamp_(0,1)
#             r_k2 = chain_mode_product(res2_up, [D4.t(), _Ek@self.apply_D[2].t(), self.apply_D[0].t(), self.apply_D[1].t()])
#             # r_k2 = chain_mode_product(res2_up, [D4.t(), _Ek@D3.t(), self.apply_D[0].t(), self.apply_D[1].t()])
#             r_k = r_k1 + r_k2
#             lmbda_ = self.lmbda[k] if self.multi_lmbda else self.lmbda
#             gamma_k = thresh_fn(gamma_k - r_k, lmbda_)
        
#         clean_spatial_cubes = chain_mode_product(gamma_k, [D4, self.apply_D[2], self.apply_D[0], self.apply_D[1]])
#         # clean_spatial_cubes = chain_mode_product(gamma_k, [D4, D3, self.apply_D[0], self.apply_D[1]])
#         # clean_spatial_cubes = chain_mode_product(PD1, [D4, self.apply_D[2], self.apply_D[0], self.apply_D[1]])

#         clean_spatial = block._agregate_blocks(clean_spatial_cubes.squeeze(0))
#         x_out = torch.matmul(clean_spatial.permute(0,2,3,1), _Ek.t()).permute(0,3,1,2).clamp_(0,1)
#         return x_out, _Ek, clean_spatial, gamma_k, D4
