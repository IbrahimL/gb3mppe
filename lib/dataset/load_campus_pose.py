import pickle
a_file = open("campuse_pose_voxel.pkl", "rb")
output = pickle.load(a_file)

a_file.close()
# l'indice 0 correspond  à l image 305
print(output[0])