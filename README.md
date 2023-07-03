# APIDFF-Net
APIDFF-Net: Adaptive Learning Point Cloud and Image Diversity Feature Fusion Network for 3D Object Detection.

**New:** The code is about to be released, and we have released our visualization results. We have also released the weight file of the model and the test results of the model, which you can find below. Anyone can upload the test results to the KITTI server to evaluate the model.

## Abstract
3D object detection is a critical task in the fields of virtual reality and autonomous driving. Given that each sensor has its own strengths and limitations, multi-sensor-based 3D object detection has gained popularity. However, most existing methods extract high-level image semantic features and fuse them with point cloud features, focusing solely on consistent information from both sensors while ignoring their complementary information. In this paper, we present a novel two-stage multi-sensor deep neural network, called the Adaptive Learning Point Cloud and Image Diversity Feature Fusion Network (APIDFF-Net), for 3D object detection. Our approach employs the fine-grained image information to complement the point cloud information by combining low-level image features with high-level point cloud features. Specifically, we design a shallow image feature extraction module to learn fine-grained information from images, instead of relying on deep layer features with coarse-grained information. Furthermore, we design a Diversity Feature Fusion (DFF) module that transforms low-level image features into point-wise image features and explores their complementary features through an attention mechanism, ensuring an effective combination of fine-grained image features and point cloud features. Experiments on the KITTI benchmark show that the proposed method outperforms state-of-the-art methods.

## Network
The overall network architecture.
![image](img/1.jpg)

## Visualization
![image](img/2.jpg)

## Pretrained model
You could download the pretrained model(Car) of APIDFF-Net from [APIDFF-Net](https://pan.baidu.com/s/1RY6nkQ6bUBUofsStHx3ZGQ?pwd=urry) which is trained on the *train* split (3712 samples) and evaluated on the *val* split (3769 samples) and *test* split (7518 samples). The verification set results are located below, and we will place the test set results on the online drive. Anyone can evaluate them through the official KITTI.
```
bbox AP:96.3399, 93.9368, 92.0453
bev  AP:95.7859, 88.9924, 88.7231
3d   AP:92.4268, 82.9832, 80.4240
aos  AP:96.26, 93.49, 91.39
```
