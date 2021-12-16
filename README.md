# DFDAFuse-pytorch
This is the code for "DFDAFuse: An Infrared and Visible Image Fusion Network Using Densely Multi-Scale Feature Extraction and Dual Attention".

## Architecture:<br>
DFDAFuse is an infrared and visible image fusion network. The current version supports Python=3.6, CUDA=11.1 and PyTorch=1.8, but it should work fine with similar versions of CUDA and PyTorch. 
<div align=center><img src="https://github.com/yhbxhhya28/DFDAFuse/imgs/model.png" width="640" height="560"/></div><br>

## Usage
######## To train:
download the source training dataset(with a total of 221 pairs) through [*Github*](https://github.com/hanna-xu/RoadScene), then crop by using the the function of'transforms.TenCrop(128,vertical_flip=False/True)' to obtain 4420 pairs images. After you have prepared the training data, please put their paths into the file, and then start the training through the following instructions:
$ cd ~/DFDAFuse
$ python3 main.py
######## To Test:
The test dataset includes [TNO](https://github.com/jianlihua123/TNO_Image_Fusion_Dataset) and [VIFB benchmark](https://github.com/xingchenzhang/VIFB). The directory 'test_image' provides all the image tested in our paper.
$ cd ~/DFDAFuse
$ python3 test.py
The results will be saved in the `./generator/test_img/` folder.

## Acknowledgement
The code of measure metrics is created based on VIFB[1]. We thank the authors of VIFB very much for making it publicly available. We also thank the contributors to the open source dataset, and all the authors who proposed SSIM, SCD, SF and Vif image evaluation methods.

We also thank all authors of the integrated images, VIF methods and evaluation metrics for sharing their work to the community!

## Contact
If you have any question, please email to me (yyyhya28@163.com).

## References
[1] Zhang X ,  Ye P ,  Xiao G . VIFB: A Visible and Infrared Image Fusion Benchmark[C]// IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. IEEE, 2020.
