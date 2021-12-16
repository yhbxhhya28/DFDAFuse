import  sys
from torch.backends import cudnn
import torch
import torch.nn as nn
from torchsummary import summary
from  strategy import  max_fuse
from thop import profile
from torchvision.models import resnet50
class SpatialAttention(nn.Module):
	def __init__(self, kernel_size=3):
		super(SpatialAttention, self).__init__()
		assert kernel_size in (3, 7), "kernel size must be 3 or 7"
		padding = 3 if kernel_size == 7 else 1
		self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		avgout = torch.mean(x, dim=1, keepdim=True)
		#print("avgout", avgout.shape)
		maxout, _ = torch.max(x, dim=1, keepdim=True)
		#print("maxout", maxout.shape)
		y = torch.cat([avgout, maxout], dim=1)
		#print("pool.shape",y.shape)  #n 2 256 256
		y = self.conv(y)
		y=self.sigmoid(y)
		#print((x*y.expand_as(x)).shape)
		return (x*y.expand_as(x)+x)

class ChannelAttention(nn.Module):
	def __init__(self, in_channels):
		super(ChannelAttention, self).__init__()
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.maxpool = nn.AdaptiveMaxPool2d(1)
		self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1, padding=0, stride=1, bias=False)
		self.relu = nn.ReLU()
		self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, padding=0, stride=1, bias=False)
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		avg_out = self.fc2(self.relu(self.fc1(self.avgpool(x))))
		#print("avg_out",self.avgpool(x).shape)
		max_out = self.fc2(self.relu(self.fc1(self.maxpool(x))))
		#print("max_out", self.maxpool(x).shape)
		y = avg_out + max_out
		y = self.sigmoid(y)
		return (x*y.expand_as(x)+x)

class eca_layer(nn.Module):
	def __init__(self, in_channels, k_size=3):
		super(eca_layer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
		self.sigmoid = nn.Sigmoid()
	def forward(self, x):
		b, c, h, w = x.shape
		y = self.avg_pool(x)
		#print("ave_pool:",y.shape)
		y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
		#print("conv:",y.shape)
		y = self.sigmoid(y)
		#print(y.shape,(x * y.expand_as(x)+x).shape)
		return (x * y.expand_as(x)+x)

class SELayer(nn.Module):
	def __init__(self, in_channels, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(in_channels, in_channels // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // reduction, in_channels, bias=False),
			nn.Sigmoid())
	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		#print(((x*y.expand_as(x)+x).shape)) # n 64 1 1
		#return x *  y.expand_as(x)
		return (x*y.expand_as(x)+x)

class Feature_extraction_denseasp(nn.Module):
	def __init__(self):
		super(Feature_extraction_denseasp, self).__init__()
		self.C1=nn.Sequential(
			nn.Conv2d(in_channels=1,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.Dasp1 = nn.Sequential(
			nn.BatchNorm2d(16),
			nn.ReLU(),
			nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1))
		# nn.LayerNorm([256, 256],elementwise_affine=False),
		# nn.ReLU(inplace=True))
		self.Dasp2=nn.Sequential(
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=2,dilation=2))
			#nn.LayerNorm([256, 256],elementwise_affine=False),
			#nn.ReLU(inplace=True))
		self.Dasp3=nn.Sequential(
			nn.BatchNorm2d(48),
			nn.ReLU(),
			nn.Conv2d(in_channels=48,out_channels=16,kernel_size=3,stride=1,padding=3,dilation=3))
			#nn.LayerNorm([256, 256],elementwise_affine=False),
			#nn.ReLU(inplace=True))
		# 	#nn.LayerNorm([256, 256],elementwise_affine=False),
		# 	#nn.ReLU(inplace=True))
	def forward(self,x):
		shallow_feature=self.C1(x)
		dasp1_feature=self.Dasp1(shallow_feature)
		dasp1_=torch.cat((dasp1_feature,shallow_feature),1)

		dasp2_feature=self.Dasp2(dasp1_)
		#dasp2_feature=self.Dasp2(dasp1_feature)
		dasp2_=torch.cat((dasp2_feature,dasp1_),1)

		dasp3_feature=self.Dasp3(dasp2_)
		#dasp3_feature = self.Dasp3(dasp2_feature)
		dasp3_out=torch.cat((dasp3_feature,dasp2_),1)

		# dasp4_feature = self.Dasp4(dasp3_out)
		# dasp_out = torch.cat((dasp4_feature, dasp3_out), 1)
		# #print("d1:",dasp1_.shape)
		# #print("d2:",dasp2_.shape)
		# #print("dasp_out.shape:",dasp_out.shape)
		return shallow_feature,dasp3_out,dasp1_,dasp2_#Edasp3_out,dasp1_,dasp2_  #out_channels: 64,shallow:16

class CAFeature_fusion(nn.Module):
	def __init__(self):
		super(CAFeature_fusion, self).__init__()
		self.CA=eca_layer(in_channels=64)
		self.SA=SpatialAttention()
	def forward(self,feature_ir,feature_vis):
		attentionmap_ir=self.CA(feature_ir)
		attentionmap_ir=self.SA(attentionmap_ir)
		attentionmap_vis=self.CA(feature_vis)
		attentionmap_vis=self.SA(attentionmap_vis)
		out=torch.cat((attentionmap_ir,attentionmap_vis),1)
		out_p=attentionmap_ir+attentionmap_vis
		#print(out.shape)
		return out#:shape:128

class Feature_reconstruction_skip(nn.Module):
	def __init__(self):
		super(Feature_reconstruction_skip, self).__init__()
		self.C2=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=64,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.C3=nn.Sequential(
			nn.Conv2d(in_channels=112,out_channels=48,   #112 48
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(48),
			nn.ReLU())
		self.C4=nn.Sequential(
			nn.Conv2d(in_channels=80,out_channels=32,  #80 32
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU())
		self.C5=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.C6 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=1,
					  kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1),
			nn.ReLU())
	def forward(self,x,d1,d2):
		x=self.C2(x)           #out 64
		x=torch.cat((x,d2),1)  #out channel: 144
		x=self.C3(x)
		x = torch.cat((x, d1), 1)  # out channel: 128
		x=self.C4(x)  #out:64
		#x=torch.cat((x,d1),1)
		x=self.C5(x)  # 16
		#x = self.C6_(x)
		x = self.C6(x)  # 1
		return x

class Feature_reconstruction(nn.Module):
	def __init__(self):
		super(Feature_reconstruction, self).__init__()
		self.C2=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=64,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU())
		self.C3=nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=32,  #64
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU())
		self.C4=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=16,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(16),
			nn.ReLU())
		self.C5=nn.Sequential(
			nn.Conv2d(in_channels=16,out_channels=1,
						kernel_size=3,stride=1,padding=1),
			nn.BatchNorm2d(1),
			nn.ReLU())

	def forward(self,x):
		x=self.C2(x)
		x=self.C3(x)
		x=self.C4(x)
		x=self.C5(x)
		return x

class DFDAFuse(nn.Module):
	def __init__(self):
		super(DFDAFuse, self).__init__()
		self.Feature_extraction_block=Feature_extraction_denseasp()
		self.Feature_reconstruction_block_skip=Feature_reconstruction_skip()
		self.CA_fusion_block=CAFeature_fusion()
	def forward(self,ir,vis):#,d1_ir,d2_ir #,d1_vis,d2_vis
		shallow_ir,ir_fext,d1_ir,d2_ir=self.Feature_extraction_block(ir)
		shallow_vis, vis_fext,d1_vis,d2_vis = self.Feature_extraction_block(vis)
		#d1_max=max_fuse(d1_ir,d1_vis)
		d1_average=(d1_ir+d1_vis)/2
		#d2_max=max_fuse(d2_ir,d2_vis)
		d2_average=(d2_vis+d2_ir)/2
		#d1_=d1_max+d1_average
		#d2_=d2_max+d2_average
		#print("ir_fext",ir_fext.shape)
		CA_fusion=self.CA_fusion_block(ir_fext,vis_fext)
		out = torch.cat((ir_fext, vis_fext), 1)
		# d1 = torch.cat((d1_max, d1_vis), 1)
		# d2 = torch.cat((d2_max, d2_vis), 1)
		#print(d1.shape,d2.shape)
		#print("CA_fusion.shape:",CA_fusion.shape)  #n 128 256 256
		img_=self.Feature_reconstruction_block_skip(CA_fusion,d1_average,d2_average)
		#img_recon = self.Feature_reconstruction_block(CA_fusion)
		return img_

####### info ######
print("python version: ",'\t',sys.version)
print("cuda   version: ",'\t',torch.version.cuda)
print("torch  version: ",'\t',torch.__version__)
print("Cnn    version: ",'\t',torch.backends.cudnn.version(),'\t',"is_availabel?:",cudnn.is_available())
print("GPU    Type:    ",'\t',torch.cuda.get_device_name(0),'\t',"GPU_nums：",torch.cuda.device_count())
print("torch.cuda.current_device():",torch.cuda.current_device())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DFDAFuse().to(device)
input1 = torch.randn(1, 1, 128 , 128).cuda()
input2 = torch.randn(1, 1, 128, 128).cuda()
macs, params = profile(model, inputs=[(input1),(input2),])
print(macs,params)