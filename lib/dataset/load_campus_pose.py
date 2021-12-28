import pickle
import os

class voxel_campus():
    def __init__ (self) :
        file=os.path.abspath('../../data/Campus/campuse_pose_voxel.pkl')
        a_file = open(file, "rb")
        self.output = pickle.load(a_file)
        a_file.close()
# l'indice 0 correspond  à l image 305
    def get_data_campus(self,frame=350,camera=0,joints='joints_3d'):
        if frame in list(range(350, 471)) + list(range(650, 751)) :
            frame-=350
            return self.output [frame*3+camera][joints]
        else :
            print('Frame not in list')
            return []
        
if __name__ == "__main__":
    print("joints 2d de l'image 400 , camera 0: ")
    print(voxel_campus().get_data_campus(400,0,'joints_2d'))
