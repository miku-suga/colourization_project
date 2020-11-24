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

## Results 

## Discussion
*lessons learned, lingering problems/limitations with your implementation, future work*
