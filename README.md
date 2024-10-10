# Regional Dynamic Point Cloud Completion Network

This repository is still under constructions.

If you have any questions about the code, please email me. Thanks!

The paper is still under review, and a pdf will be provided after subsequent review.

A brief introduction to RD-Net:
We propose a novel shape completion network, RD-Net, which innovatively focuses on information interaction between points to provide local and global information for generating fine-grained complete shapes. (Specific content will be added later)

##0) Environment
Pytorch 1.0.1
Python 3.7.4

##1) Dataset
```
  cd dataset
  bash download_shapenet_part16_catagories.sh 
```
##2) Train
```
python Train_RDNet.py 
```
‘crop_point_num’ : control the number of missing points.
‘point_scales_list ’ : control different input resolutions.

##3) Evaluate the Performance on ShapeNet
```
python show_recon_RDNet.py
```
Show the completion results, the program will generate txt files in 'test example'.
```
python show_CD.py
```
Show the Chamfer Distances and two metrics in our paper.

##4) Visualization of Examples

Using Meshlab to visualize  the txt files.
