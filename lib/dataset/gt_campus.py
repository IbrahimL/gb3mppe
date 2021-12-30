import pandas as pd
import tensorflow as tf
import os

'''
GroundTruth Campus 
Les données sont traitées à partir de l'image 704
frame 1 ====> image 704

'''
class gt():
    def __init__(self):
        col=["cam1 : x","cam1 : y","cam2 : x","cam2 : y","cam3 : x","cam3 : y"]
        cwd = os.path.abspath('../../data/Campus/')
        # Données 2D
        self.data_actor1= pd.read_csv(cwd+'/personne1.txt',sep='\s+',names=col)
        self.data_actor2= pd.read_csv(cwd+'/personne2.txt',sep='\s+',names=col)
        self.data_actor3= pd.read_csv(cwd+'/personne3.txt',sep='\s+',names=col)
        self.data=pd.concat([self.data_actor1,self.data_actor2,self.data_actor3], axis=1).to_numpy()
        # Données 3D
        col3D=['x','y','z']
        self.data3D_actor1= pd.read_csv(cwd+'/personne1_3D.txt',sep='\s+',names=col3D)
        self.data3D_actor2= pd.read_csv(cwd+'/personne2_3D.txt',sep='\s+',names=col3D)
        self.data3D_actor3= pd.read_csv(cwd+'/personne3_3D.txt',sep='\s+',names=col3D) 
        self.data3D=pd.concat([self.data3D_actor1,self.data3D_actor2,self.data3D_actor3], axis=1).to_numpy()
    # renvoie les coordonnées 2D d'une articulation , dans une vue donnée 
    def get_coord(self,frame=1,actor=1,camera=1):
        pose2D=self.data[(frame-1)*14:(frame-1)*14+14,(actor-1)*6:(actor-1)*6+6][:,(camera-1)*2:(camera-1)*2+2]
        return pose2D
    # renvoie les coord 2D du centre humain
    def get_human_center (self,frame=1,actor=1,camera=1):
        right_hip=self.get_coord(frame,actor,camera)[2]
        left_hip=self.get_coord(frame,actor,camera)[3]
        return (right_hip+left_hip)/2
    
    # renvoie les coordonnées 3D d'une articulation
    def get_coord3D(self,frame=1,actor=1):
        pose3D=self.data3D[(frame-1)*14:(frame-1)*14+14,(actor-1)*3:(actor-1)*3+3]
        return pose3D
    # renvoie les coord 3D du centre humain
    def get_human_center3D (self,frame=1,actor=1):
        right_hip=self.get_coord3D(frame,actor)[2]
        left_hip=self.get_coord3D(frame,actor)[3]
        return (right_hip+left_hip)/2
    
if __name__ == "__main__":
    
    frame=2
    actor=3
    camera=1
    hc=gt().get_human_center(frame,actor,camera)
    print("Human center 2D: ", hc)
    hc3D=gt().get_human_center3D(frame,actor)
    print("Human center 3D: ", hc3D)
