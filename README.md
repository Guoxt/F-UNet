# F-UNet: Deep convolutional neural network based on feature engineering for medical image segmentation
Referring to the idea of Feature Engineering in traditional machine learning, we propose a deep convolution neural network based on Feature Engineering(F-UNet). The key of F-UNet is to alleviate the scarcity of annotated data by constructing the diversity of feature engineering. F-UNet is divided into two stages. In the first stage, a basic segmentation network is trained to build more diverse feature engineering. In the second stage, the improved network is trained by the diversity feature engineering constructed in the first stage.
# Paper
This repository provides the official Pytorch implementation of F-UNet in the following papers:

F-UNet: Deep convolutional neural network based on feature engineering for medical image segmentation  
Xutao Guo, Ting Ma  
Harbin Institute of Technology at Shenzhen  
# Folder
* `data`:the folder where dataset is placed.  
* `models`:model files.  
* `check`:utils files(include many utils)     
  * `main`:training, validation and test function.  
  * `sets`:some configuration about project parameters.  
# Prerequisites
* PyTorch 1.0  
  `conda install torch torchvision`   
* tqdm  
  `conda install tqdm`  
* imgaug  
  `conda install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`  
  `conda install imgaug`  
