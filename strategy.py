import collections
import cv2
import math
import os
import numpy as np
import torch
import random
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import  utils
import torch.nn.functional as F
from NR_IQA.brisquequality import *

def max_fuse(source_a, source_b):
    dimension = source_a.shape
    torch_a = source_a.detach().view(-1)
    torch_b = source_b.detach().view(-1)
    torch_max=torch.max(torch_a, torch_b)
    torch_max=torch_max.view(dimension[0],dimension[1],dimension[2],-1)
    #print(torch_max,torch_max.shape)
    return torch_max

def save_image( image, path, name,part):
        t = image.data
        t[t > 1] = 1
        t[t < 0] = 0
        utils.save_image(t, "%s/%s_%d.jpg" % (path, name,part))

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def entropy_1d(img):
    img=(img.squeeze(0))*255.0
    img=img.cpu().numpy()
    h,w=img.shape
    hist_cv = cv2.calcHist([img], [0], None, [256], [0.0,255.0])
    P = hist_cv / ( h * w )  # 概率
    E = np.sum([p * np.log2(1 / (p + 1e-10)) for p in P])
    # plt.subplot(111)
    # plt.plot(hist_cv)
    # plt.show()
    return  E

def entropy_2d(img):
    img=(img.squeeze(0))*255.0
    size=img.shape[0]*img.shape[1]
    tuple=[]
    en=0.0
    #img = img.astype(np.float32)
    window_k = torch.Tensor([[[[1, 1, 1], [1, 0, 1], [1, 1, 1]]]]).cuda()
    imgd4_ =img.unsqueeze(0).unsqueeze(0)
    out = (F.conv2d(imgd4_, window_k, stride=1, padding=1)//8).squeeze(0).squeeze(0)
    #print(img.shape,out.shape)
    for i, (im, o) in enumerate(zip(img, out)):
        for im_,o_ in zip(im,o):
            tuple.append((im_.item(),o_.item()))
    dict_= collections.Counter(tuple)
    for key,value in dict_.items():
        en +=(value/size) * math.log2(1 / ((value/size) + 1e-10))
    return  en

def brisque(img):
    img = (img.squeeze(0)) * 255.0
    img_arr=img.detach().cpu().numpy()
    nr_iqa=1/test_measure_BRISQUE(img_arr)
    return nr_iqa

def gen_gaussian_noise(signal,SNR,bs):
    signal=signal.cuda()
    # for i in range(bs):
    for i in range(bs):
        signal_=signal[i,:,:,:]
        noise=torch.randn(*signal_.shape).cuda()
        noise=noise-torch.mean(noise)
        signal_power=(1/(signal_.shape[0]*signal_.shape[1]*signal_.shape[2]))*torch.sum(torch.pow(signal_,2))
        noise_variance=signal_power/torch.pow(torch.tensor(10),(SNR/10))
        noise=(torch.sqrt(noise_variance)/torch.std(noise))*noise
        if (i<1):
            noise_bs=noise
        if (i==1):
            noise_bs=torch.stack((noise_bs,noise))
        if (i>1):
            noise_bs = torch.cat((noise_bs, noise.unsqueeze(0)))
    return noise_bs.cuda()

def check_snr(signal,noise,bs):
    SNR=[]
    for i in range(bs):
        signal_=signal[i,:,:,:]
        noise_=noise[i,:,:,:]
        signal_power = (1 / (signal_.shape[0] * signal_.shape[1] * signal_.shape[2])) * torch.sum(torch.pow(signal_, 2))
        noise_power = (1 / (noise_.shape[0] * noise_.shape[1] * noise_.shape[2])) * torch.sum(torch.pow(noise_, 2))  # 0.90688
        SNR.append(10*torch.log10(signal_power/noise_power))
    return SNR
