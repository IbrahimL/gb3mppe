import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

class MLP(Model):
    '''
    '''
    def __init__(self, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_hid1 = layers.Dense(hidden_dim, activation='relu')
        self.linear_hid2 = layers.Dense(hidden_dim, activation='relu')
        self.linear_out = layers.Dense(output_dim, activation='relu')
    
    def call(self, input):
        if len(input.shape) == 1:
            input = tf.expand_dims(input, axis=0)
        x = self.linear_hid1(input)
        x = self.linear_hid2(x)
        x = self.linear_out(x)
        return x

class EdgeConv(Layer):
    def __init__(self, h_theta):
        '''
        '''
        super(EdgeConv, self).__init__()
        self.h_theta = h_theta

    def _aggregate(self, inputs):
        '''
        '''
        x_v, x_vp = inputs
        return self.h_theta(tf.concat([x_v, x_vp-x_v], -1))

    def _prep_features(self, Adjacency, node_features):
        '''
        '''
        A_temp = tf.cast(Adjacency, tf.int32) - tf.eye(Adjacency.shape[-1], dtype=tf.int32)
        Deg = tf.reduce_sum(A_temp, axis=0)
        Deg = tf.cast(Deg, tf.int32)
        Feat_rep_n = tf.repeat(node_features, Deg, axis=0)
        Feat_rep_v = tf.gather(node_features, tf.where(A_temp==1)[:,1])
        return Feat_rep_n, Feat_rep_v, Deg

    def _convolution(self, inputs):
        '''
        '''
        Adjacency, node_features = inputs
        Feat_rep_n, Feat_rep_v, Deg = self._prep_features(Adjacency, node_features)
        res = tf.squeeze(tf.vectorized_map(self._aggregate, (Feat_rep_n, Feat_rep_v)))
        max_id = tf.vectorized_map(tf.norm, res)
        res = tf.split(res, Deg)
        max_id = tf.split(max_id, Deg)
        get_ind = lambda x: tf.math.argmax(x, axis=-2)
        max_id = tf.vectorized_map(get_ind, max_id)
        res = tf.ragged.stack(res)
        res_tensor = tf.gather(res, max_id, batch_dims=1)
        return res_tensor
    
    def call(self, Adjacency, node_features):
        inputs = (Adjacency, node_features)
        if len(Adjacency.shape) == 2:
            return self._convolution(inputs)
        return tf.vectorized_map(self._convolution, inputs)

class EdgeConvE(Layer):
    '''
    '''
    def __init__(self, params):
        super(EdgeConvE, self).__init__()

    def build(self):
        pass

    def call(self, Adjacency, node_features, edge_attributes):
        pass

if __name__ == "__main__":
    # Test MLP class
    h_dim = 4
    o_dim = 4
    mlp = MLP(h_dim, o_dim)
    x = tf.random.uniform((10, 80))
    # Test EdgeConv
    ec = EdgeConv(mlp)
    A = tf.convert_to_tensor([[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]])
    Feat = tf.convert_to_tensor([[1, 2],[2, 3], [4, 5], [5, 6]])
    print("test 1:")
    print(ec(A, Feat))
    # Test EdgeConv (Batch)
    ec = EdgeConv(mlp)
    A = tf.convert_to_tensor([[[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]], 
                  [[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]]])
    Feat = tf.convert_to_tensor([[[1, 2],[2, 3], [4, 5], [5, 6]],
                     [[1, 2],[2, 3], [4, 5], [5, 6]]])
    print("test 2")
    print(ec(A, Feat))