import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from load import  *
from batch_transformers import *
from model import  *
from loss import  *
from  strategy import  *

EPS = 1e-8

class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed) 
        self.train_vis_transform = transforms.Compose([
            BatchToResize(),
            BatchToTensor(),   
        ])
        self.train_ir_transform = transforms.Compose([
            #BatchToPILImage(),
            BatchToResize(),
            BatchToTensor(),
            #BatchRGBToYCbCr()
        ])
        self.test_vis_transform = transforms.Compose([
            #BatchToPILImage(),
            BatchToTensor(),
            #BatchRGBToYCbCr()
        ])
        self.test_ir_transform = self.test_vis_transform

        self.train_batch_size =config.train_batch_size
        self.test_batch_size = config.test_batch_size
        # training set configuration
        self.train_data = ImageSeqDataset(vis_file=os.path.join(config.trainset, 'vis_train.txt'),
                                          ir_file=os.path.join(config.trainset, 'ir_train.txt'),
                                          vis_transform=self.train_vis_transform,
                                          ir_transform=self.train_ir_transform)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size= self.train_batch_size,
                                       shuffle=False,
                                       pin_memory=True,  #或加快速度
                                       num_workers=8)
        #initialize the model
        self.model = DFDAFuse()
        self.model_name = type(self.model).__name__
        print(self.model)
        # loss function
        self.loss_mse = L2()
        self.loss_msssim=SSIM(data_range=1.0)
        self.initial_lr = config.lr   #0.0001
        self.weight=config.weight
        if self.initial_lr is None:
            lr = 0.0005
        else:
            lr = self.initial_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        if torch.cuda.is_available() and config.use_cuda:
            self.model.cuda()
            self.loss_mse =self.loss_mse.cuda()
            self.loss_msssim=self.loss_msssim.cuda()
        # some states
        self.start_epoch = 0
        self.total_loss_every_epoch = []
        self.pixel_loss_every_epoch=[]
        self.msssim_loss_every_epoch=[]
        self.test_results = []
        self.w=[]
        self.pixel_loss_temp=0
        self.msssim_loss_temp=0
        self.use_cuda = config.use_cuda
        self.use_noise=config.noise
        self.max_epochs = config.max_epochs
        self.epochs_per_eval = config.epochs_per_eval
        self.epochs_per_save = config.epochs_per_save
        self.ckpt_path = config.ckpt_path

        #try load the model
        if config.resume or not config.train:
            if config.ckpt:
                ckpt = os.path.join(config.ckpt_path, config.ckpt)
            else:
                ckpt = self._get_latest_checkpoint(path=config.ckpt_path)
            self._load_checkpoint(ckpt=ckpt)

        self.scheduler = lr_scheduler.StepLR(self.optimizer,
                                             last_epoch=self.start_epoch-1,
                                             step_size=config.decay_interval,
                                             gamma=config.decay_ratio)

    def start(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            _ = self._train_single_epoch(epoch)

    def _train_single_epoch(self, epoch):
        # initialize logging system
        batches = len(self.train_loader)
        iter_count = epoch * batches + 1
        start_time = time.time()
        # start training
        print('Adam learning rate: {:f}'.format(self.optimizer.param_groups[0]['lr']),'\t',"batches:",batches)
        for _, sample_batched in enumerate(self.train_loader, 0):
            #TODO: remove this after debugging
            I_vis, I_ir = sample_batched['I_vis'], sample_batched['I_ir']
            #print(I_vis.shape,I_ir.shape)
            I_vis = torch.squeeze(I_vis, dim=1)
            I_ir = torch.squeeze(I_ir, dim=1)
            Y_vis = I_vis[:, 0, :, :].unsqueeze(1)
            Y_ir = I_ir[:, 0, :, :].unsqueeze(1)   #yyy: NCHW
            if self.use_cuda:
                Y_vis = Y_vis.cuda()
                Y_ir = Y_ir.cuda()
            Y_ir_=Y_ir
            Y_vis_=Y_vis
            if(self.use_noise):
                guassi_noise_i = gen_gaussian_noise(Y_ir, 30,self.train_batch_size)
                guassi_noise_v = gen_gaussian_noise(Y_vis_, 30, self.train_batch_size)
                Y_ir_=Y_ir+guassi_noise_i
                Y_ir_ = Y_ir_.data
                Y_ir_[Y_ir_ > 1] = 1
                Y_ir_[Y_ir_ < 0] = 0
            self.optimizer.zero_grad()
            bs_fused= self.model(Y_ir_, Y_vis_)
            #print("bs_fused/shape:",bs_fused.shape,bs_fused.shape[0])
            pixel_loss_temp = 0
            ssim_value=0
            w_ir=[]

            for fused in range(bs_fused.shape[0]):
                if((len(self.w)<batches)&(len(w_ir)<=self.train_batch_size)):
                    en_ir = entropy_2d(Y_ir[fused,:,:,:])
                    en_vis = entropy_2d(Y_vis[fused,:,:,:])
                    iqa_ir=brisque(Y_ir[fused,:,:,:])
                    iqa_vis=brisque(Y_vis[fused, :, :, :])
                    #print(en_ir,en_vis,iqa_vis,iqa_ir)
                    if (iqa_vis<0 or iqa_vis>0.1):
                        iqa_vis=0
                        iqa_ir=0
                    w_ir.append((en_ir+100*iqa_ir)/(en_ir+en_vis+100*iqa_ir+100*iqa_vis))
                    if (w_ir[fused]<0.3 or w_ir[fused]>0.65):
                        w_ir[fused]=0.5
                    weight_ir=w_ir[fused]
                #print(weight_ir)
                else:
                    l = iter_count % batches-1
                    if(l==-1):
                        l=batches-1
                    weight_ir = self.w[l][fused]
                    #print(weight_ir)
                pixel_loss_temp +=weight_ir*self.loss_mse(bs_fused[fused,:,:,:].unsqueeze(0), Y_vis[fused,:,:,:].unsqueeze(0))
                pixel_loss_temp +=(1-weight_ir)*self.loss_mse(bs_fused[fused, :, :, :].unsqueeze(0),Y_ir[fused, :, :, :].unsqueeze(0))
                ssim_value  += (1-weight_ir)*self.loss_msssim(bs_fused[fused,:,:,:].unsqueeze(0), Y_vis[fused,:,:,:].unsqueeze(0))
                ssim_value += weight_ir* self.loss_msssim(bs_fused[fused, :, :, :].unsqueeze(0),Y_ir[fused, :, :, :].unsqueeze(0))
            if iter_count < (batches+1):
                self.w.append(w_ir)
            print(len(self.w),len(self.w[-1]))
            pixel_loss_value = pixel_loss_temp / bs_fused.shape[0]
            msssim_loss_value =1-(ssim_value / bs_fused.shape[0])
            total_loss=20*pixel_loss_value+msssim_loss_value
            format_str = ('(E:%d/bhy200,bs:%d,lr:%f,current_iter_num:%d) [total_loss= %f,pixel_loss= %f,msssim_loss= %f]')
            print(format_str % (epoch,self.train_batch_size,self.optimizer.param_groups[0]['lr'],iter_count,total_loss,pixel_loss_value,msssim_loss_value))
            total_loss.backward()
            self.optimizer.step()
            self.pixel_loss_temp +=pixel_loss_value
            self.msssim_loss_temp +=msssim_loss_value
            if (iter_count) % 20 == 0:
                self.pixel_loss_every_epoch.append(self.pixel_loss_temp / 20)
                self.msssim_loss_every_epoch.append(self.msssim_loss_temp / 20)
                self.total_loss_every_epoch.append(( self.msssim_loss_temp + self.pixel_loss_temp) / 20)
                format_str = ('(E:%d times:%dth) [average_20iter_total_loss= %f]')
                print(format_str % (epoch,iter_count/20,self.total_loss_every_epoch[-1]))
                self.pixel_loss_temp = 0.
                self.msssim_loss_temp =0.
            iter_count += 1
            end_time = time.time()
            current_epoch_time=end_time-start_time
        self.scheduler.step()
        if (epoch+1) % self.epochs_per_save == 0:
            model_name = '{}-{:0>5d}.pt'.format(self.model_name, epoch)
            model_name = os.path.join(self.ckpt_path, model_name)
            self._save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'train_loss': np.array(self.total_loss_every_epoch),
                'pixel_loss':np.array(self.pixel_loss_every_epoch),
                'msssim_loss':np.array(self.msssim_loss_every_epoch),
                'self-w':np.array(self.w),
                'test_results': self.test_results,
            }, model_name)
        '''
        if (epoch+1) % self.epochs_per_eval == 0: 
            # evaluate after every other epoch
            eva_results_all,eva_results_pixel,eva_results_gra = self.eval(epoch)
            self.test_results.append(eva_results_all)
            out_str = 'Epoch {} Testing: Average total-loss: {:.4f}'.format(epoch, eva_results_all)
            print(out_str)
        '''
        return total_loss

    '''
    def eval(self, epoch):
        scores_all = []
        scores_pixel = []
        scores_gra = []
        scores_msssim = []
        for step, sample_batched in enumerate(self.test_loader, 0):
            # TODO: remove this after debugging
            I_vis, I_ir = sample_batched['I_vis'], sample_batched['I_ir']
            I_vis = torch.squeeze(I_vis, dim=1)
            I_ir = torch.squeeze(I_ir, dim=1)

            Y_vis = I_vis[:, 0, :, :].unsqueeze(1)
            Y_ir = I_ir[:, 0, :, :].unsqueeze(1)
            Cb_vis = I_vis[:, 1, :, :].unsqueeze(1)
            Cr_vis = I_vis[:, 2, :, :].unsqueeze(1)

            Wb = (torch.abs(Cb_vis - 0.5) + EPS) / torch.sum(torch.abs(Cb_vis - 0.5) + EPS, dim=0)
            Wr = (torch.abs(Cr_vis - 0.5) + EPS) / torch.sum(torch.abs(Cr_vis - 0.5) + EPS, dim=0)
            Cb_f = torch.sum(Wb * Cb_vis, dim=0, keepdim=True).clamp(0, 1)
            Cr_f = torch.sum(Wr * Cr_vis, dim=0, keepdim=True).clamp(0, 1)

            if self.use_cuda:
                Y_vis = Y_vis.cuda()
                Y_ir = Y_ir.cuda()
            eval_fused = self.model(Y_ir, Y_vis)  #1 1 h w
            for ef_tensor in eval_fused:
                eval_pixel_loss = self.loss_mse(ef_tensor, Y_ir)
                total_loss=eval_pixel_loss
                O_syne_RGB = YCbCrToRGB()(torch.cat((ef_tensor.cpu(), Cb_f, Cr_f), dim=1))
            #q = self.loss_fn(O_syne_RGB, I_vis).cpu()
                scores_all.append(total_loss)
                scores_pixel.append(eval_pixel_loss)
                self._save_image(O_syne_RGB, self.fused_img_path, str(epoch) + '_' + str(step))
                self._save_image(ef_tensor, self.y_map_path, str(epoch) + '_' + str(step))
        avg_quality_all = sum(scores_all) / len(scores_all)
        avg_quality_pixel=sum(scores_pixel) / len(scores_pixel)
        avg_quality_gra = sum(scores_gra) / len(scores_gra)
        return avg_quality_all,avg_quality_pixel,avg_quality_gra
    '''
    def _load_checkpoint(self, ckpt):
        if os.path.isfile(ckpt):
            print("[*] loading checkpoint '{}'".format(ckpt))
            checkpoint = torch.load(ckpt)
            self.start_epoch = checkpoint['epoch']+1
            self.train_loss = checkpoint['train_loss']
            self.test_results = checkpoint['test_results']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.w=(checkpoint['self-w'])
            if self.initial_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt))

    @staticmethod
    def _get_latest_checkpoint(path):
        ckpts = os.listdir(path)
        ckpts = [ckpt for ckpt in ckpts if not os.path.isdir(os.path.join(path, ckpt))]
        all_times = sorted(ckpts, reverse=True)
        return os.path.join(path, all_times[0])

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        # if os.path.exists(filename):
        #     shutil.rmtree(filename)
        torch.save(state, filename)

    def _save_image(self, image, path, name):
        b = image.size()[0]
        for i in range(b):
            t = image.data[i]

            t[t > 1] = 1
            t[t < 0] = 0

            utils.save_image(t, "%s/%s_%d.png" % (path, name, i))
