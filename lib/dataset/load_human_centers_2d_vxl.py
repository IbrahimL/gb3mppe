import pickle
import os

file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/Campus/voxel_2d_human_centers.pkl')
a_file = open(file, "rb")
human_centers = pickle.load(a_file)
a_file.close()

'''
Les images vont de 704 à 1998

'''
# exemple
print('Voxel pose \n',human_centers['image_704'])
# la ya trois personnes d'où les 3 vecturs (x,y) pr chacune des cameras



file_GT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/Campus/GT_2d_human_centers.pkl')
a_file = open(file_GT, "rb")
human_centers_GT = pickle.load(a_file)
a_file.close()

'''
Les images vont de 704 à 1998

'''
# exemple
print('Ground Truth \n',human_centers_GT['image_704'])
# la ya trois personnes d'où les 3 vecturs (x,y) pr chacune des cameras