import numpy as np
import tensorflow as tf
import pickle, os
import numpy as np

class Campus:
    def __init__(self, feature_path):
        super().__init__()
        
        self.feature_path = feature_path
        self.max_n_persons = 3
        self.n_cameras = 3
        self.n_samples = 1295
    
    def _get_feat_dict(self):
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.feature_path)
        with open(load_path, "rb") as file:
            data = pickle.load(file)
        camera_0 = [value for key, value in data.items() if 'c0' in key.lower()]
        camera_1 = [value for key, value in data.items() if 'c1' in key.lower()]
        camera_2 = [value for key, value in data.items() if 'c2' in key.lower()]
        self.n_samples = len(camera_0)
        return camera_0, camera_1, camera_2
    
    def _generate_graphs(self):
        cameras = self._get_feat_dict()
        node_features = np.zeros([self.n_samples, self.max_n_persons*self.n_cameras, 512])
        adjacency = np.ones([self.n_samples, self.max_n_persons*self.n_cameras, self.max_n_persons*self.n_cameras])
        edge_features = np.zeros([self.n_samples, self.max_n_persons*self.n_cameras, self.max_n_persons*self.n_cameras, 1])
        for i in range(self.n_samples):
            for j in range(self.n_cameras):
                for k, feats in enumerate(cameras[j][i]):
                    node_features[i, k, :] = feats[0]
        node_features = tf.convert_to_tensor(node_features)
        adjacency = tf.convert_to_tensor(adjacency)
        edge_features = tf.convert_to_tensor(edge_features)
        return node_features, adjacency, edge_features
        

if __name__ == "__main__":
    campus_dataset = Campus('../../data/Campus/node_features.pkl')
    camera_0, camera_1, camera_2 = campus_dataset._get_feat_dict()
    node_features, adjacency, edge_features = campus_dataset._generate_graphs()
    print(node_features.shape)
    print(node_features[0, 0, :])

    
