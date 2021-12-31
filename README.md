# gb3mppe

This is a re-implementation for:

# [Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images](https://arxiv.org/abs/2109.05885)
Size Wu, Sheng Jin, Wentao Liu, Lei Bai, Chen Qian, Dong Liu, Wanli Ouyang
2021 

# Installation

1. Clone this repo, and we'll call the directory that you cloned multiview-multiperson-pose as ${gb3mppe}.
2. Install dependencies.

# Data preparation for MMG
We train and evaluate our model on the **[Campus](http://campar.in.tum.de/Chair/MultiHumanPose)**.
1. We processed the ground-truth and the 2D pose estimated by [Hanyue Tu, Chunyu Wang, Wenjun Zeng](https://github.com/microsoft/voxelpose-pytorch) to our format.
2. We also created dataset in form of python dictionnaries (pkl files) to store the node features and the edge features which ar the inputs of our MMG model.
3. You can download all these data from [here](https://drive.google.com/drive/folders/1Ck5ireXtLGGKFdgb5UJQe1spikul_O0K) and place it in gb3mppe/data/campus.
4. You can also try to generate the node and edge features by runing [generate_mmg_features.py](https://github.com/IbrahimL/gb3mppe/blob/12dd1b94396ccc0328c1b8a08882a0de45ec954d/lib/utils/generate_mmg_features.py) and [generate_edge_features.py](https://github.com/IbrahimL/gb3mppe/blob/12dd1b94396ccc0328c1b8a08882a0de45ec954d/lib/utils/generate_edge_features.py) resp. , the node features are extracted from feature maps that ar constructed with [2D pose estimator trained on COCO](https://github.com/microsoft/voxelpose-pytorch). These feature maps are the output of the two last deconv layers of the prerained PoseResnet.


The directory tree should look like this:

