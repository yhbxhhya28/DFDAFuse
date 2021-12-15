import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class L21_res(nn.Module):
    def __init__(self):
        super(L21_res, self).__init__()

    def forward(self,img1,img2):
        h_img = img1.size()[-2]
        w_img = img1.size()[-1]
        l21_res=0
        for i in range(h_img):
            #x=torch.pow((img1[:,:,i,:]-img2[:,:,i,:]),2)
            x1 = torch.mean(torch.pow((img1[:,:,i,:]-img2[:,:,i,:]),2))
            x2 = torch.mean(torch.pow((img1[:, :, :, i] - img2[:, :, :, i]), 2))
            #print(x)
            l21_res +=x2+x1
        out=l21_res/w_img
        return out

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()

    def forward(self,img1,img2):
        return torch.mean(torch.pow((img1-img2),2))

class gradient_1d(nn.Module):
    def __init__(self):
        super(gradient_1d, self).__init__()
    def forward(self,img):
        # gradient step=1
        l = img
        r = F.pad(img, [0, 1, 0, 0])[:, :, :, 1:]
        t = img
        b = F.pad(img, [0, 0, 0, 1])[:, :, 1:, :]
        dx, dy = torch.abs(r - l), torch.abs(b - t)
        # dx will always have zeros in the last column, r-l
        # dy will always have zeros in the last row,    b-t
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0
        d_img=dx+dy
        return  d_img

class gradient_2d(nn.Module):
    def __init__(self):
        super(gradient_2d, self).__init__()
    def forward(self, img):
        kernel =  torch.Tensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]).cuda()
        kernel =  kernel.unsqueeze(0).unsqueeze(0)
        out = F.conv2d(img, kernel, stride=1, padding=1).squeeze(0).squeeze(0)
        return out

class gra_dif(nn.Module):
    def __init__(self):
        super(gra_dif,self).__init__()
    def forward(self, img1, img2):
        #gradient_model = gradient_1d().cuda()
        gradient_model = gradient_2d().cuda()
        h_img = img1.size()[-2]
        w_img = img1.size()[-1]
        img1_gra = gradient_model(img1)
        img2_gra = gradient_model(img2)
        gra_dif = pow(torch.norm(img1_gra - img2_gra), 2) / (h_img*w_img)
        return gra_dif

def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2
    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)
def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out
def _ssim(X, Y, win, data_range=1.0, size_average=True, full=False, K=(0.01,0.03), nonnegative_ssim=True):
    K1, K2 = K
    batch, channel, height, width = X.shape
    compensation = 1.0
    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2
    win = win.to(X.device, dtype=X.dtype)
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2) # set alpha=beta=gamma=1
    if nonnegative_ssim:
        cs_map = F.relu( cs_map, inplace=True )
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    if nonnegative_ssim:
        ssim_map = F.relu( ssim_map, inplace=True )
    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
         ssim_val = ssim_map#.mean(-1).mean(-1).mean(-1)  # reduce along CHW
         cs = cs_map#.mean(-1).mean(-1).mean(-1)
    #print(ssim_val.mean())
    if full:
        return ssim_val, cs,mu2
    else:
        return cs
def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=1.0, size_average=False, full=True, K=(0.01, 0.03), nonnegative_ssim=True):
    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')
    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')
    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')
    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]
    ssim_val, cs ,m= _ssim(X, Y,win=win,data_range=data_range,size_average=False,full=True, K=K, nonnegative_ssim=nonnegative_ssim)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()
    if full:
        return ssim_val, cs,m
    else:
        return cs
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1.0, size_average=False, channel=1, K=(0.01, 0.03), nonnegative_ssim=True):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
    def forward(self, X, Y):
        ssim_value,c_value,m_value= ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
        #print(ssim_value.mean())
        return  ssim_value.mean()

def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=None, size_average=True, full=False, weights=None, K=(0.01, 0.03), nonnegative_ssim=False):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')
    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')
    smaller_side = min( X.shape[-2:] )
    # 异常触发
    assert smaller_side > (win_size-1) * (2**4), \
         "Image size should be larger than %d due to the 4 downsamplings in ms-ssim"% ((win_size-1) * (2**4))
    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)
    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]
    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs,m = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=True,
                             full=True, K=K, nonnegative_ssim=nonnegative_ssim)
        mcs.append(cs)
        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)
    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val_Cs = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1)))  # (batch, )
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )
    if size_average:
        msssim_val = msssim_val.mean()
        msssim_val_Cs=msssim_val_Cs.mean()
    return msssim_val
class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=1, weights=None, K=(0.01, 0.03), nonnegative_ssim=True):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative to avoid NaN results.
        """
        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim
        self.win_size = win_size
        self.win_sigma = win_sigma
    def forward(self, X, Y):
        (_, channel, _, _) = Y.size()
        win_ =  self.win
        s=ms_ssim(X, Y, win=win_, size_average=self.size_average, data_range=self.data_range, weights=self.weights, K=self.K, nonnegative_ssim=self.nonnegative_ssim)
        return s
'''
ssim1=SSIM(data_range=255)
x=torch.linspace(1,196,196).view(1,1,14,14).repeat(1,1,1,1)
y=x*2
z_s=ssim1(x,y)
print(z_s)
'''
