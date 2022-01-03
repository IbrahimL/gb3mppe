# gb3mppe

This is a re-implementation for:

# [Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images](https://arxiv.org/abs/2109.05885)
Size Wu, Sheng Jin, Wentao Liu, Lei Bai, Chen Qian, Dong Liu, Wanli Ouyang
2021 

# Installation

1. Clone this repo, and we'll call the directory that you cloned multiview-multiperson-pose as ${gb3mppe}.
2. Install dependencies.
# MMG : Multi-view Matching Graph Module
## Data preparation for MMG
We train and evaluate our model on the **[Campus](http://campar.in.tum.de/Chair/MultiHumanPose)**.
1. We processed the ground-truth and the 2D pose estimated by [Hanyue Tu, Chunyu Wang, Wenjun Zeng](https://github.com/microsoft/voxelpose-pytorch) to our format.
2. We also created dataset in form of python dictionnaries (pkl files) to store the node features and the edge features which ar the inputs of our MMG model.
3. You can download all these data from [here](https://drive.google.com/file/d/1ZeyehIh8TZkR6qAaUngoF3hdtTBpm9UK/view?usp=sharing) and place it in gb3mppe/data/campus.
4. You can also try to generate the node and edge features by runing [generate_mmg_features.py](https://github.com/IbrahimL/gb3mppe/blob/12dd1b94396ccc0328c1b8a08882a0de45ec954d/lib/utils/generate_mmg_features.py) and [generate_edge_features.py](https://github.com/IbrahimL/gb3mppe/blob/12dd1b94396ccc0328c1b8a08882a0de45ec954d/lib/utils/generate_edge_features.py) resp. , the node features are extracted from feature maps that ar constructed with [2D pose estimator trained on COCO](https://github.com/microsoft/voxelpose-pytorch). These feature maps are the output of the two last deconv layers of the prerained PoseResnet.


The directory tree should look like this:

```
${gb3mppe}
|-- data
    |-- campus
        |-- campuse_pose_voxel.pkl
        |-- cfg.yaml
        |-- Camera4
        |-- GT_2d_human_centers.pkl
        |-- GT_3d_human_centers.pkl
        |-- node_features.pkl
        |-- personne1.txt
        |-- personne1_3D.txt
        |-- personne2.txt
        |-- personne2_3D.txt
        |-- personne3.txt
        |-- personne3_3D.txt
        |-- voxel_2d_human_centers.pkl
        |-- voxel_3d_human_centers.pkl
        |-- CampusSeq1
            |-- Camera0
            |-- Camera1
            |-- Camera2
            |-- actorsGT.mat
            |-- calibration_campus.json
            |-- pred_campus_maskrcnn_hrnet_coco.pkl
|-- lib
    |-- dataset
    |-- gt_coord_2D
    |-- models
    |-- utils              
|-- test
```
here's the [link]( http://campar.cs.tum.edu/files/belagian/multihuman/CampusSeq1.tar.bz2 )  to donwload CampusSeq1 .


# CRG : Center Refinement Graph

We also implementred the CRG's architecture, but unfortunately, we did not have enough time to generate the node and edge features to train the model. However, you can test the implementation on a dummy dataset that we created, where inputs have the same shape than the real features.
To do that, you can run this [notebook test](https://github.com/IbrahimL/gb3mppe/blob/8020d4bda261a91b7053d0b524e5b80bb7b3815f/test/test_architectures_fake_data.ipynb)

# PRG : Pose Regression Graph
Like CRG, you can test the PRG's implementation on a dummy dataset that we created, where inputs have the same shape than the real features.
To do that, you can run the [same notebook](https://github.com/IbrahimL/gb3mppe/blob/8020d4bda261a91b7053d0b524e5b80bb7b3815f/test/test_architectures_fake_data.ipynb)
