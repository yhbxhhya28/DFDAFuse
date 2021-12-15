import os
import functools
import torch
import pandas as pd
from PIL import Image
import  cv2
from torch.utils.data import Dataset
from batch_transformers import  BatchToTensor, BatchRGBToYCbCr,BatchToPILImage, YCbCrToRGB,BatchToResize
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import  numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
	filename_lower = filename.lower()
	return any(filename_lower.endswith(ext) for ext in extensions)

def image_path_loader(img_per_path):
	img_per_path= os.path.expanduser(img_per_path)
	every_img_path = []
	if has_file_allowed_extension(img_per_path, IMG_EXTENSIONS):
		'''
		image_bgr = cv2.imread(img_per_path)
		image_bgr2rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		every_img_path.append(image_bgr2rgb)
		'''
		pil_im=Image.open(img_per_path).convert('RGB')
		every_img_path.append(Image.open(img_per_path).convert('RGB'))
	return every_img_path
def get_default_every_img_path():
	return functools.partial(image_path_loader)
class ImageSeqDataset(Dataset):
	def __init__(self,vis_file=None,
				 ir_file=None,
				 vis_transform=None,
				 ir_transform=None,
				 img_loader=get_default_every_img_path):
		self.vis_seqs = pd.read_csv(vis_file, sep='\n', header=None)
		self.ir_seqs = pd.read_csv(ir_file, sep='\n', header=None)
		self.vis_transform = vis_transform
		self.ir_transform = ir_transform
		self.load_per_img=img_loader()
	def __getitem__(self, index):
		vis_path=self.vis_seqs.iloc[index, 0]
		ir_path=self.ir_seqs.iloc[index,0]
		vis_per= self.load_per_img(vis_path)
		ir_per=self.load_per_img(ir_path)
		if self.vis_transform is not None:
			vis = self.vis_transform(vis_per)
		if self.ir_transform is not None:
			ir = self.ir_transform(ir_per)
		I_vis = torch.stack(vis, 0).contiguous()
		I_ir = torch.stack(ir, 0).contiguous()
		sample = {'I_vis': I_vis, 'I_ir': I_ir}
		return sample

	def __len__(self):
		return len(self.vis_seqs)

	@staticmethod
	def _reorderBylum(seq):
		I = torch.sum(torch.sum(torch.sum(seq, 1), 1), 1)
		_, index = torch.sort(I)
		result = seq[index, :]
		return result


