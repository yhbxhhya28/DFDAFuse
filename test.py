import time
from load import  *
from batch_transformers import *
from model import  *
from loss import  *
from main import  parse_config
from strategy import  *

def tensor2array(image_tensor, imtype=np.uint8, normalize=False):
      image_numpy = image_tensor.cpu().float().numpy()
      if normalize:
          image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
      else:
          image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
      image_numpy = np.clip(image_numpy, 0, 255)
      if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
          image_numpy = image_numpy[:, :, 0]
      return image_numpy.astype(imtype)

def save_image(img, save_dir, name, Gray=True): #True for the RGB output, False for the grayscale
    fext="bmp"
    imgPath = os.path.join(save_dir, "%s_%s.%s" % (name,"DFDA_" ,fext))
    img_array =tensor2array(img.data)
    image_pil = Image.fromarray(img_array)
    if Gray:
        image_pil.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        image_pil.save(imgPath)   #  #

def main(config):
    torch.manual_seed(config.seed)
    test_vis_transform = transforms.Compose([
        # BatchToPILImage(),
        BatchToTensor(),
        BatchRGBToYCbCr()
    ])
    test_ir_transform = test_vis_transform
    test_batch_size = config.test_batch_size
    for ijk in range(1):
        test_data = ImageSeqDataset(vis_file=os.path.join(config.testset, "sample_vis.txt"),
                                         ir_file=os.path.join(config.testset, "sample_ir.txt"),
                                         vis_transform=test_vis_transform,
                                         ir_transform=test_ir_transform)
        test_loader = DataLoader(test_data,
                                      batch_size=test_batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=12)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        f_extract=Feature_extraction_denseasp()
        f_fuse=CAFeature_fusion()
        f_recon=Feature_reconstruction_skip()
        model = DFDAFuse()
        model.to(device)
        _mse = L2().cuda()
        _mmssim = SSIM(data_range=1.0).cuda()
        print(len(os.listdir(config.ckpt_path)))
        for i ,ckpt_path in enumerate(sorted(os.listdir(config.ckpt_path)),0):
            save_pt=os.path.join(config.test_fused_result)#,str(i)+"/"+img_dir[ijk_])
            ckpt_pt=os.path.join(config.ckpt_path,ckpt_path)
            if not os.path.exists(save_pt):
                 os.makedirs(save_pt)
            print("[*] loading checkpoint '{}'".format(ckpt_pt))
            checkpoint = torch.load(ckpt_pt)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            test_mse=0
            test_msssim=0
            test_total=0

            with torch.no_grad():
                Time = []
                for step, sample_batched in enumerate(test_loader, 1) :
                    start_time = time.time()
                    I_vis, I_ir = sample_batched['I_vis'], sample_batched['I_ir']
                    I_vis = torch.squeeze(I_vis, dim=1)
                    I_ir = torch.squeeze(I_ir, dim=1)
                    Y_vis = I_vis[:, 0, :, :].unsqueeze(1)
                    Y_ir = I_ir[:, 0, :, :].unsqueeze(1)
                    Cb_vis = I_vis[:, 1, :, :].unsqueeze(1).cuda()
                    Cr_vis = I_vis[:, 2, :, :].unsqueeze(1).cuda()
                    # Wb = (torch.abs(Cb_vis - 0.5) + EPS) / torch.sum(torch.abs(Cb_vis - 0.5) + EPS, dim=0)
                    # Wr = (torch.abs(Cr_vis - 0.5) + EPS) / torch.sum(torch.abs(Cr_vis - 0.5) + EPS, dim=0)
                    # Cb_f = torch.sum(Wb * Cb_vis, dim=0, keepdim=True).clamp(0, 1)
                    # Cr_f = torch.sum(Wr * Cr_vis, dim=0, keepdim=True).clamp(0, 1)
                    Y_vis = Y_vis.cuda()
                    Y_ir = Y_ir.cuda()

                    test_fused = model(Y_ir, Y_vis)  # 1 1 h w
                    test_mse += _mse(test_fused, Y_ir)
                    test_msssim +=0.5*(_mmssim(test_fused,Y_vis))+0.5*(_mmssim(test_fused,Y_ir))
                    test_total = test_mse + test_msssim

                    test_fused = YCbCrToRGB()(torch.cat((test_fused, Cb_vis, Cr_vis), dim=1)).squeeze(0)
                    save_image(test_fused, save_pt, str(step))
                    current_time = time.time()
                    Time.append(current_time-start_time)
                print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))
                print(test_total.data.item()/21,test_mse.data.item()/21,test_msssim.data.item()/21)

if __name__ == '__main__':
    config = parse_config()
    if not os.path.exists(config.test_fused_result):
        os.makedirs(config.test_fused_result)
    main(config)
