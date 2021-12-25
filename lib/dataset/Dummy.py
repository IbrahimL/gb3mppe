from numpy.core.defchararray import upper
import tensorflow as tf
import numpy as np

class Dummy:
    def __init__(self, n_samples, n_nodes, node_feat_dim, edge_feat_dim, self_loops=True):
        super().__init__()
        
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.self_loops = self_loops
        
    def _get_adjacency(self):
        # generate random adjacency matrices
        adj_mats = np.random.randint(low=0, high=2, size=[self.n_samples, self.n_nodes, self.n_nodes], dtype=int)
        upper_tr = np.triu(adj_mats, k=1)
        adj_mats = upper_tr + np.transpose(upper_tr, axes=(0,2,1))
        if self.self_loops:
            adj_mats += np.eye(self.n_nodes, dtype=int)
        output =  tf.convert_to_tensor(adj_mats, dtype=tf.float32)
        return tf.sparse.from_dense(output)
    
    @tf.function
    def _get_node_features(self):
        node_feat_mats = tf.random.uniform(shape=[self.n_samples, self.n_nodes, self.node_feat_dim],
                                      minval=0, maxval=1, dtype=tf.float32)
        return node_feat_mats
    
    @tf.function
    def _get_edge_features(self):
        edge_feat_mats = tf.random.uniform(shape=[self.n_samples, self.n_nodes, self.n_nodes, self.edge_feat_dim],
                                      minval=0, maxval=1, dtype=tf.float32)
        return edge_feat_mats

    def _get_db(self):
        adj_mats = self._get_adjacency()
        n_feat_mats = self._get_node_features()
        e_feat_mats = self._get_edge_features()
        db = {"adjacency": adj_mats,
              "node_features": n_feat_mats,
              "edge_features": e_feat_mats}
        return db
        
    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    dummy = Dummy(2, 4, 8, 1)
    dummy_dataset = dummy._get_db()
    print(dummy_dataset['adjacency'])
    print('------------')
    print(dummy_dataset['node_features'])
    print('------------')
    print(dummy_dataset['edge_features'])