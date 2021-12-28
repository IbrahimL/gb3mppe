import pickle
import os

file = os.path.abspath('../../data/Campus/voxel_2d_human_centers.pkl')
a_file = open(file, "rb")
human_centers = pickle.load(a_file)
a_file.close()

'''
Les images vont de 704 à 1998

'''
# exemple
print(human_centers['image_704'])
# la ya trois personnes d'où les 3 vecturs (x,y) pr chacune des cameras