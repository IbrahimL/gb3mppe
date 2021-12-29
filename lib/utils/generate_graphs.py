import torch
from torch import Tensor
from poseresnet import *
import yaml
import os, pickle
from typing import Dict, Iterable, Callable
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

class dicttoclass:
    def __init__(self, dic):
        for key, val in dic.items():
            if type(val) is dict:
                val = dicttoclass(val)
            setattr(self, key, val)

def get_coords(path='../../data/Campus/voxel_2d_human_centers.pkl'):
    file = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    a_file = open(file, "rb")
    human_centers = pickle.load(a_file)
    a_file.close()
    return human_centers

def get_features(featureExtractor, images, upsample_size, coords):
    B = images.shape[0]
    H, W = upsample_size
    output_deconv1 = torch.zeros([B, 256, H, W])
    output_deconv2 = torch.zeros([B, 256, H, W])
    features = featureExtractor(images)
    out_1 = features["deconv_layers.5"]
    out_2 = features["deconv_layers.8"]
    print(coords)
    for i in range(B):
        output_deconv1[i, :, :, :] = torch.nn.Upsample(size=[H, W])(out_1)
        output_deconv2[i, :, :, :] = torch.nn.Upsample(size=[H, W])(out_2)
    return output_deconv1[:, :, int(coords[1]), int(coords[0])], output_deconv2[:, :, int(coords[1]), int(coords[0])]

def main():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../data/Campus/cfg.yaml')
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../data/Campus/CampusSeq1')
    cfg = yaml.safe_load(open(cfg_path))
    cfg = dicttoclass(cfg)
    coords = get_coords()
    model = get_pose_net(cfg, False)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    posenet_features = FeatureExtractor(model, layers=['deconv_layers.5','deconv_layers.8'])
    for frame, poses in coords.items():
        for camera, pose in poses.items():
            zeros = "00000"
            _, frame_number = frame.split("_")
            frame_number_str = zeros[:5-len(frame_number)] + str(frame_number)
            path_img = os.path.join(data_path, camera[:-2] + camera[-1], 'campus4-c'+camera[-1]+'-'+frame_number_str+".png")
            input_image = Image.open(path_img)
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)
            for coords in pose:
                output_deconv1, output_deconv2 = get_features(posenet_features, input_batch, [288, 360], coords)
                features = torch.cat([output_deconv1, output_deconv2], dim=-1)
                print(features.shape)
            

if __name__ == "__main__":
    main()
    
    
        
