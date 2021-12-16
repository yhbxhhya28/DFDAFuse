import scipy.io as sio
import numpy as  np
import matplotlib.pyplot as plt
import os
import pandas as pd

'''
#tno#
plane = sio.loadmat('./results_TNO.mat')
print(plane["resultsMetrics"].shape)
SSIM=plane["resultsMetrics"][:,:,0]
SCD=plane["resultsMetrics"][:,:,1]
SF=plane["resultsMetrics"][:,:,2]
VIF=plane["resultsMetrics"][:,:,3]
'''

#VIFB#
plane = sio.loadmat('./results_VIFB.mat')
print(plane["resultsMetrics"].shape)
SSIM=plane["resultsMetrics"][:,:,0]
SCD=plane["resultsMetrics"][:,:,1]
SF=plane["resultsMetrics"][:,:,2]
VIF=plane["resultsMetrics"][:,:,3]


plt.figure(figsize=(32,28))
labels = ['GTF','DRTV','FusionGAN','GANMcC','DRF','DDcGAN','U2Fusion','Ours']
plt.subplot(221)
plt.title('SSIM',fontsize=28)
f_ssim=plt.boxplot(SSIM,vert = True,labels=labels,patch_artist=True,showmeans=True,showfliers=False,meanprops = {'markerfacecolor':'yellow',"markersize":12})
plt.grid(linestyle="--", alpha=0.3)
color = ['c', 'c', 'c', 'c','c', 'c', 'deepskyblue', 'r']
for box, c in zip(f_ssim['boxes'], color):
    box.set(color=c, linewidth=5)
    box.set(facecolor=c)
for median in f_ssim['medians']:
    median.set(color='black', linewidth=5)
    plt.xticks(size=20)
    plt.yticks(fontsize=19)

plt.subplot(222)
plt.title('SCD',fontsize=28)
f_scd=plt.boxplot(x = SCD,labels=labels,vert = True,patch_artist=True,showmeans=True,showfliers=False,meanprops = {'markerfacecolor':'yellow',"markersize":12})
plt.grid(linestyle="--", alpha=0.3)
color = ['c', 'c', 'c', 'c','c', 'c', 'r', 'deepskyblue']
for box, c in zip(f_scd['boxes'], color):
    box.set(color=c, linewidth=5)
    box.set(facecolor=c)
for median in f_scd['medians']:
    median.set(color='black', linewidth=5)
    plt.xticks(size=20)
    plt.yticks(fontsize=19)

plt.subplot(223)
plt.title('SF',fontsize=28)
f_sf=plt.boxplot(x = SF,vert = True,labels=labels,patch_artist=True,showmeans=True,showfliers=False,meanprops = {'markerfacecolor':'yellow',"markersize":12})
plt.grid(linestyle="--", alpha=0.3)
color = ['c', 'c', 'c', 'c','c', 'r', 'c', 'deepskyblue']
for box, c in zip(f_sf['boxes'], color):
    box.set(color=c, linewidth=5)
    box.set(facecolor=c)
for median in f_sf['medians']:
    median.set(color='black', linewidth=5)
    plt.xticks(size=20)
    plt.yticks(fontsize=19)

plt.subplot(224)
plt.title('VIF',fontsize=28)
f_vif=plt.boxplot(x = VIF,vert = True,labels=labels,patch_artist=True,showmeans=True,showfliers=False,meanprops = {'markerfacecolor':'yellow',"markersize":12})
plt.grid(linestyle="--", alpha=0.3)
color = ['c', 'c', 'c', 'deepskyblue','c', 'c', 'c', 'r']
for box, c in zip(f_vif['boxes'], color):
    box.set(color=c, linewidth=2)
    box.set(facecolor=c)
for median in f_vif['medians']:
    median.set(color='black', linewidth=5)
    plt.xticks(size=20)
    plt.yticks(fontsize=19)
plt.savefig(fname="metrics.PDF")
plt.show()
