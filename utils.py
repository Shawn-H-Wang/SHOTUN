# Author: JianJun Liu
# Date: 2022-1-13
import numpy as np
import scipy.io as sio
import os
import torch
import cv2
import torch.nn.functional as fun
import torch.utils.data as data
import pytorch_ssim


class toolkits(object):
    @staticmethod
    def compute_psnr(img1: np.ndarray, img2: np.ndarray, channel=True):
        assert img1.ndim == 3 and img2.ndim == 3
        img_h, img_w, img_c = img1.shape
        ref = img1.reshape(-1, img_c)
        tar = img2.reshape(-1, img_c)
        msr = np.mean((ref - tar) ** 2, 0)
        if channel is False:
            max2 = np.max(ref) ** 2  # channel-wise ???
        else:
            max2 = np.max(ref, axis=0) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean
    
    @staticmethod
    def compute_psnr_gpu(img1, img2):
        c, w, h = img1.shape
        ref = img1.reshape(c, -1)
        tar = img2.reshape(c, -1)
        msr = torch.mean((ref - tar) ** 2, 1)
        max1 = torch.max(ref, 1)[0]
        psnrall = 10 * torch.log10(max1**2 / msr)
        out_mean = torch.mean(psnrall)
        return out_mean

    @staticmethod
    def compute_sam(label: np.ndarray, output: np.ndarray):
        h, w, c = label.shape
        x_norm = np.sqrt(np.sum(np.square(label), axis=-1))
        y_norm = np.sqrt(np.sum(np.square(output), axis=-1))
        xy_norm = np.multiply(x_norm, y_norm)
        xy = np.sum(np.multiply(label, output), axis=-1)
        dist = np.mean(
            np.arccos(np.minimum(np.divide(xy, xy_norm + 1e-8), 1.0 - 1.0e-9))
        )
        dist = np.multiply(180.0 / np.pi, dist)
        return dist
    
    @staticmethod
    def compute_sam_gpu(x_true, x_pred):
        w, h, c = x_true.shape
        x_true = x_true.reshape(-1, c)
        x_pred = x_pred.reshape(-1, c)
        
        # x_pred[torch.where((torch.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

        sam = (x_true * x_pred).sum(axis=1) / (
            torch.linalg.norm(x_true, 2, 1) * torch.linalg.norm(x_pred, 2, 1)
        )

        sam = torch.arccos(sam) * 180 / torch.pi
        mSAM = sam.mean()
        # var_sam = torch.var(sam)
        return mSAM

    @staticmethod
    def check_dir(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def channel_last(input_tensor: np.ndarray, squeeze=True):
        if squeeze is True:
            input_tensor = np.squeeze(input_tensor)
        input_tensor = np.transpose(input_tensor, axes=(1, 2, 0))
        return input_tensor

    @staticmethod
    def channel_first(input_tensor: np.ndarray, expand=True):
        input_tensor = np.transpose(input_tensor, axes=(2, 0, 1))
        if expand is True:
            input_tensor = np.expand_dims(input_tensor, axis=0)
        return input_tensor


class torchkits(object):
    @staticmethod
    def extract_patches(input_tensor: torch.Tensor, kernel=3, stride=1, pad_num=0):
        # input_tensor: N x C x H x W, patches: N * H' * W', C, h, w
        if pad_num != 0:
            input_tensor = torch.nn.ReflectionPad2d(pad_num)(input_tensor)
        all_patches = input_tensor.unfold(2, kernel, stride).unfold(3, kernel, stride)
        N, C, H, W, h, w = all_patches.shape
        all_patches = all_patches.permute(0, 2, 3, 1, 4, 5)
        all_patches = torch.reshape(all_patches, shape=(N * H * W, C, h, w))
        return all_patches

    @staticmethod
    def torch_norm(input_tensor: torch.Tensor, mode=1):
        if mode == 1:
            loss = torch.sum(torch.abs(input_tensor))
            return loss
        return None
    
    @staticmethod
    def torch_ssim_loss(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor, mode=1):
        if mode == 1:
            ssim_loss = pytorch_ssim.SSIM()
            loss = 1 - ssim_loss(input_tensor1, input_tensor2)
            return loss
        return None

    @staticmethod
    def get_param_num(model):
        num = sum(x.numel() for x in model.parameters())
        print("model has {} parameters in total".format(num))
        return num

    @staticmethod
    def to_numpy(val: torch.Tensor):
        return val.cpu().detach().numpy()


class BlurDownOri(object):
    def __init__(self, shift_h=0, shift_w=0, stride=0):
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf, pad, groups, ratio):
        if psf.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1)
        if self.stride == 0:
            output_tensor = fun.conv2d(
                input_tensor, psf, None, (1, 1), (pad, pad), groups=groups
            )
            output_tensor = output_tensor[
                :, :, self.shift_h :: ratio, self.shift_h :: ratio
            ]
        else:
            output_tensor = fun.conv2d(
                input_tensor, psf, None, (ratio, ratio), (pad, pad), groups=groups
            )
        return output_tensor



class BlurDown(object):
    def __init__(self, shift_h=0, shift_w=0, stride=0):
        self.shift_h = 0
        self.shift_w = 0
        self.stride = 0
        pass

    def __call__(self, input_tensor: torch.Tensor, psf1, psf2, pad, groups, ratio):
        psf = psf1@psf2
        if psf1.shape[0] == 1:
            psf = psf.repeat(groups, 1, 1, 1)
            # psf1 = psf1.repeat(groups, 1, 1, 1)
            # psf2 = psf2.repeat(groups, 1, 1, 1)
        if self.stride == 0:
            # output_tensor = fun.conv2d(
            #     input_tensor, psf1, None, (1, 1), (pad, 0), groups=groups
            # )
            output_tensor = fun.conv2d(
                input_tensor, psf, None, (1, 1), (pad, pad), groups=groups
            )
            output_tensor = output_tensor[
                :, :, self.shift_h :: ratio, self.shift_h :: ratio
            ]
        else:
            output_tensor = fun.conv2d(
                input_tensor, psf1.repeat(groups, 1, 1, 1), None, (ratio, 1), (pad, 0), groups=groups
            )
            output_tensor = fun.conv2d(
                output_tensor, psf2.repeat(groups, 1, 1, 1), None, (1, ratio), (0, pad), groups=groups
            )
        return output_tensor
    
class BlurUp(object):
    def __init__(self, shift_h=0, shift_w=0, stride=0):
        self.shift_h = shift_h
        self.shift_w = shift_w
        self.stride = stride
        pass

    def __call__(self, input_tensor: torch.Tensor, psf1, psf2, pad, groups, ratio):
        if psf1.shape[0] == 1:
            psf1 = psf1.repeat(groups, 1, 1, 1)
            psf2 = psf2.repeat(groups, 1, 1, 1)
        
        output_tensor = fun.conv_transpose2d(input_tensor, psf1, None, (ratio, 1), (0, 0), groups=groups)
        output_tensor = fun.conv_transpose2d(output_tensor, psf2, None, (1, ratio), (0, 0), groups=groups)
        return output_tensor


class DataInfo(object):
    """
    file structure
    ./data/
    ../data/pavia/
    ../data/moffett/
    ../data/dc/
    .../data/pavia/XXX/
    .../data/pavia/BlindTest/
    .../data/pavia/pavia_data_r?_?_?.mat
    ..../data/pavia/BlindTest/r?_?_?/
    ..../data/pavia/BlindTest/model/
    ..../data/pavia/BlindTest/BR.mat
    """

    def __init__(self, ndata=0, nratio=4, nsnr=0, data_iter=1):
        name = self.__class__.__name__
        print("%s is running" % name)
        self.gen_path = "./data/"  # change
        self.folder_names = ["pavia/"]
        """Remote Sensing Images"""
        self.data_names = ["paviac_256_s8"]
        self.noise = ["0"]
        self.file_path = (
            self.gen_path
            + self.folder_names[ndata]   
            + self.data_names[ndata]
            + ".mat"
        )
        mat = sio.loadmat(self.file_path)
        hsi, msi = mat["HSI"], mat["MSI"]  # h x w x L, H x W x l
        if len(hsi.shape) == 2:
            hsi = np.reshape(hsi, [hsi.shape[0],hsi.shape[1],1])
        if "REF" in mat.keys():
            ref = mat["REF"]  # H x W X L
        else:
            ref = np.ones(shape=(msi.shape[0], msi.shape[1], hsi.shape[2]))
        if "K" in mat.keys():
            psf, srf = mat["K"], mat["R"]  # K x K, l X L
        else:
            psf = np.ones(
                shape=(msi.shape[0] // hsi.shape[0], msi.shape[1] // hsi.shape[1])
            )
            srf = np.ones(shape=(msi.shape[-1], hsi.shape[-1]))
        self.save_path = (
            self.gen_path
            + self.folder_names[ndata]
            + name
            + str(nratio)
            + self.noise[nsnr]
            + "/"
        )
        hsi = hsi.astype(np.float32)
        msi = msi.astype(np.float32)
        self.ref = ref.astype(np.float32)
        self.psf = psf.astype(np.float32)
        self.srf = srf.astype(np.float32)
        self.model_save_path = self.save_path + "model/"
        # preprocess
        self.hsi = toolkits.channel_first(hsi)  # 1 x L x h x w
        self.msi = toolkits.channel_first(msi)  # 1 x l x H x W
        self.hs_bands, self.ms_bands = self.hsi.shape[1], self.msi.shape[1]
        self.ratio = nratio # int(self.msi.shape[-1] / self.hsi.shape[-1])
        self.height, self.width = self.msi.shape[2], self.msi.shape[3]
        pass


class PatchDataset(data.Dataset):
    def __init__(
        self,
        hsi: torch.Tensor,
        msi: torch.Tensor,
        hsi_up: torch.Tensor,
        kernel,
        stride,
        ratio=1,
    ):
        super(PatchDataset, self).__init__()
        self.hsi = torchkits.extract_patches(
            hsi, kernel // ratio, stride // ratio, pad_num=0 // ratio
        )
        self.msi = torchkits.extract_patches(msi, kernel, stride, pad_num=0)
        self.hsi_up = torchkits.extract_patches(hsi_up, kernel, stride, pad_num=0)
        self.num = self.msi.shape[0]
        assert self.hsi.shape[0] == self.num

    def __getitem__(self, item):
        hsi = self.hsi[item, :, :, :]
        msi = self.msi[item, :, :, :]
        hsi_up = self.hsi_up[item, :, :, :]
        return hsi, msi, hsi_up, item

    def __len__(self):
        return self.num
    

class CubeDataset(data.Dataset):

    def __init__(self,
                 msi_cubes: torch.Tensor,
                 hsi: torch.Tensor):
        super(CubeDataset, self).__init__()

        self.msi_cubes = msi_cubes
        self.hsi = hsi

    def __getitem__(self, item):
        msi_cube = self.msi_cubes[item,:,:,:]
        hsi = self.hsi[0,:,:,:]
        return msi_cube, hsi

    def __len__(self):
        return len(self.msi_cubes)
