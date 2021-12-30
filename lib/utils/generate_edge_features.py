#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 20:27:32 2021

@author: spi-2017-12 , Karim
"""

import tensorflow as tf
import os
import numpy as np
from epipolar_geo import *
import pickle

# lecture
file = os.path.join(os.path.dirname(os.path.abspath('epipolar_geo.py')), 
                    '../../data/Campus/voxel_2d_human_centers.pkl')
#   file = os.path.abspath('../../data/Campus/voxel_2d_human_centers.pkl')
a_file = open(file, "rb")
human_centers = pickle.load(a_file)
a_file.close()

def save_edge_features(save_path,save=False):
    feature_dic={}
    if save  :
        # calcul et ecriture des edges attributes
        for frames,camera in human_centers.items() :
            print(frames)
            test = human_centers[frames]
            dic={}
            for camera,poses in test.items():  
                edges=[]
                dic_cam={}
                nodes_num=0
                for pose in poses :
                    nodes_num+=1
                    edge1=[]
                    for camdif,posesdif in test.items() :
                        if posesdif is not poses:
                            pt1=tf.constant(pose,shape=pose.shape,dtype=tf.float32)
                            pt2=tf.constant(posesdif,shape=posesdif.shape,dtype=tf.float32)                
                            camera1=int(camera[-1])
                            camera2=int(camdif[-1])
                            edge1=np.concatenate((edge1,
                                           epipolar_geometry().edge_score(pt1,pt2,camera1,camera2,10).numpy()),
                                           axis=None)
                    edges=np.concatenate((edges,edge1),axis=None)
                    dic_cam.update({'node_'+str(nodes_num) : edge1})
        
                dic.update({camera : dic_cam})
            feature_dic.update({frames : dic})    
            # A changer biensur
            a_file = open(save_path+'edge_features.pkl', "wb")
            pickle.dump(feature_dic, a_file)
            a_file.close()   
    else :
        file = os.path.join(os.path.dirname(os.path.abspath('generate_edge_features.py')), 
                        '../../data/Campus/edge_features.pkl')
        #   file = os.path.abspath('../../data/Campus/voxel_2d_human_centers.pkl')
        a_file = open(file, "rb")
        feature_dic = pickle.load(a_file)
        a_file.close()
    return feature_dic
if __name__ == "__main__":

    # open edge feature
    
    edge_features=save_edge_features(save_path='',save=False)
    # les edge features de la premiere image par exemple , camera 0
    print(edge_features['image_704']['camera_0'])
