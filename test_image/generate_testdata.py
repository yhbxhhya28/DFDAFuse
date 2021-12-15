####### generates the test data ######
import os


####### Please replace your own path of test imageset. #######
out_dir="/home/zt/deep_learning/DL_yhb/Image_Fusion/DFDAFuse/test_image/"

irimg_vifbdir="/home/zt/deep_learning/DL_yhb/Image_Fusion/DFDAFuse/test_image/VIFB/ir/"    
vimg_vifbdir="/home/zt/deep_learning/DL_yhb/Image_Fusion/DFDAFuse/test_image/VIFB/vis/"

irimg_tnodir="/home/zt/deep_learning/DL_yhb/Image_Fusion/DFDAFuse/test_image/TNO/ir/"
vimg_tnodir="/home/zt/deep_learning/DL_yhb/Image_Fusion/DFDAFuse/test_image/TNO/vis/"

for i in range(2):
    if(i<=0):
        num = 0
        img_path =os.path.join(os.path.expanduser(irimg_vifbdir))
        save_txt=os.path.expanduser(out_dir+"VIFB_ir")
    else:
        img_path = os.path.join(os.path.expanduser(vimg_vifbdir))
        save_txt=os.path.expanduser(out_dir+"VIFB_vis")
    print(img_path)
    for root, dir, fnames in sorted(os.walk(img_path)):
        for fname in sorted(fnames,key=lambda x: int(x.split('.jpg')[0])): # 'bmp' for TNO dataset, 'jpg' for VIFB dataset.
            num=num+1
            portion = os.path.splitext(fname)
            if(num%1==0):
                with open(save_txt+".txt", "a+") as f:
                    f.write("%s\n" % (os.path.join(img_path,fname)))
