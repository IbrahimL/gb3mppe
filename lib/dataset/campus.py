import numpy as np
import tensorflow as tf
import pickle, os
import numpy as np

class Campus:
    def __init__(self, node_features_path, edge_features_path, gt_path):
        super().__init__()
        
        self.node_features_path = node_features_path
        self.edge_features_path = edge_features_path
        self.gt_path = gt_path
        self.max_n_persons = 3
        self.n_cameras = 3
        self.n_samples = 1295
    
    def _get_node_feat_dict(self):
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.node_features_path)
        node_features = np.zeros([self.n_samples, self.max_n_persons*self.n_cameras, 512])
        with open(load_path, "rb") as file:
            data = pickle.load(file)
        for frame, dict_0 in data.items():
            frame_n = int(frame) - 704
            for camera, dict_1 in dict_0.items():
                _, camera_n = camera.split("_")
                camera_n = int(camera_n)
                for node, feat in dict_1.items():
                    _, node_n = node.split("_")
                    node_n = int(node_n) - 1
                    node_features[frame_n, camera_n*self.n_cameras + node_n, :] = feat
        return node_features
                    
    
    def _get_edge_feat_dict(self):
        # Works for images between 704 and 1133
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.edge_features_path)
        edge_features = np.zeros([self.n_samples, self.max_n_persons*self.n_cameras, self.max_n_persons*self.n_cameras, 1])
        adjacency = np.zeros([self.n_samples, self.max_n_persons*self.n_cameras, self.max_n_persons*self.n_cameras])
        with open(load_path, "rb") as file:
            data = pickle.load(file)
        for frame, dict_0 in data.items():
            _, frame_n = frame.split("_")
            i = int(frame_n) - 704
            for camera, dict_1 in dict_0.items():
                _, camera_n = camera.split("_")
                j = int(camera_n) - 1
                for node, feature in dict_1.items():
                    _, node_n = node.split("_")
                    k = int(node_n) - 1
                    for l, f in enumerate(feature):
                        if j == 0:
                            edge_features[i, j*self.n_cameras + k, self.n_cameras + l, 0] = f
                            adjacency[i, j*self.n_cameras + k, self.n_cameras + l] = 1
                            adjacency[i, self.n_cameras + l, j*self.n_cameras + k] = 1
                        if j == 1:
                            edge_features[i, j*self.n_cameras + k, (2*self.n_cameras + l)%9, 0] = f
                            adjacency[i, j*self.n_cameras + k, (2*self.n_cameras + l)%9] = 1
                            adjacency[i, (2*self.n_cameras + l)%9, j*self.n_cameras + k] = 1
                        if j == 2:
                            edge_features[i, j*self.n_cameras + k, l, 0] = f
                            adjacency[i, j*self.n_cameras + k, l] = 1
                            adjacency[i, l, j*self.n_cameras + k] = 1
        return edge_features, adjacency
    
    def _generate_gt_graphs(self):
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.gt_path)
        edge_features = np.zeros([self.n_samples, self.max_n_persons*self.n_cameras, self.max_n_persons*self.n_cameras, 1])
        with open(load_path, "rb") as file:
            data = pickle.load(file)
        for frame, dict_0 in data.items():
            _, frame_n = frame.split("_")
            i = int(frame_n) - 704
            for camera, dict_1 in dict_0.items():
                _, camera_n = camera.split("_")
                j = int(camera_n) - 1
                for node, feature in dict_1.items():
                    _, node_n = node.split("_")
                    k = int(node_n) - 1
                    for l, f in enumerate(feature):
                        if j == 0:
                            edge_features[i, j*self.n_cameras + k, self.n_cameras + l, 0] = f
                        if j == 1:
                            edge_features[i, j*self.n_cameras + k, (2*self.n_cameras + l)%9, 0] = f
                        if j == 2:
                            edge_features[i, j*self.n_cameras + k, l, 0] = f
        return edge_features
    
    def _generate_graphs(self):
        node_features = self._get_node_feat_dict()
        edge_features, adjacency = self._get_edge_feat_dict()
        node_features = tf.convert_to_tensor(node_features)
        gt_graphs = self._generate_gt_graphs()
        adjacency = tf.convert_to_tensor(adjacency)
        edge_features = tf.convert_to_tensor(edge_features)
        return node_features, adjacency, edge_features, gt_graphs
        

if __name__ == "__main__":
    campus_dataset = Campus('../../data/Campus/node_features.pkl', '../../data/Campus/edge_features.pkl', '../../data/Campus/GT_graphs.pkl')
    node_features, adjacency, edge_features, gt_graphs = campus_dataset._generate_graphs()
    print(node_features.shape)
    print(adjacency.shape)
    print(edge_features.shape)
    print(gt_graphs.shape)

    
