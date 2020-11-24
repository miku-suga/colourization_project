# colourization_project

# Team Kakomami Deep Learning CS1470 Final Project

Kaki So (jsu15), Kota Soda (ksoda), Miku Suga (msuga), March Boonyapaluk (kboonyap)

## Technologies Used

Python

## Overview

This colorization project implements a paper called Gray2ColorNet. 

P. Lu, J. Yu, X. Peng, Z. Zhao, X. Wang. 2020. Gray2ColorNet: Transfer More
Colors from Reference Image. In Proceedings of the 28th ACM International
Conference on Multimedia (MM ’20), October 12–16, 2020, Seattle, WA, USA.
ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3394171.3413594

## Introduction 
*what problem you’re solving and why it’s important*

This project focuses on methods to color images using a reference image. This model could be applied to many disciplines–mostly artistic, such as coloring old monochrome film videos frame by frame using pictures as reference. See [link](https://www.youtube.com/watch?v=cyL6wWOHxUM&ab_channel=IgnacioL%C3%B3pez-Francos) for where we got our inspiration. We hope that this technology will be useful for historians to restore old images for educational and research purposes. 

## Methodology
*dataset, model architecture, etc.*

### Dataset
For our project, we used images from the ImageNet database. For training, we needed a pair of colored images, T_ab and R_ab, where T is the monochromatic image stripped to its luminance channel L, and R is the reference image. Although we input T_l and R_ab into the model, T_ab is still used as a ground truth in the training process. 

### Model Architecture
The model architecture is split into two main sub networks. GCFN (Gated color fusion sub-network) fuses the semantic and color distribution information in the reference image, and MCN uses transpose convolution in order to color the monochrome image using information gathered from the GCFN. 

#### GCFN
The GCFN model is also split into two main parts, the Semantic Color assignment module and the Color distribution module. The output of the two modules are passed into the Gated fusion module, where it outputs the fused color feature M. The color distribution module uses convolution and spacial replication to output 3 feature matrices of different sizes from the histogram of the reference image. The semantic assignment module applies max pooling on the monochrome image to get the image input's class label G, applies concatnation and correlation function on the two feature matrices to get the correlation matrix C, and applies convolution on C, as well as the reference image in order to output three color feature matrices of different sizes. Finally, the gated fusion module takes in all six color feature matrices, as well as the correlation matrix (with convolution applied), and passes in the inputs through three gates in order to produce the fused color features M_1, M_2, and M_3. Each gate takes in two color features of the same size, as well as the downsized correlation matrix.   

#### MCN

## Results 

## Discussion
*lessons learned, lingering problems/limitations with your implementation, future work*
