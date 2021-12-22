from functools import reduce
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow_graphics.geometry.convolution.graph_convolution import edge_convolution_template
from tensorflow.python.ops.gen_math_ops import Add

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

    def _aggregate(self, x_v, x_vp):
        '''
        '''
        return self.h_theta(tf.concat([x_v, x_vp-x_v], -1))
    
    def call(self, Adjacency, node_features):
        return edge_convolution_template(data=node_features, 
                                         neighbors=Adjacency, 
                                         sizes=None,
                                         edge_function=self._aggregate,
                                         reduction='max',
                                         edge_function_kwargs={},
                                         name='edge_conv')
            


class EdgeConvE(Layer):
    '''
    '''
    def __init__(self, params):
        super(EdgeConvE, self).__init__()
        
    def _aggregate(self, inputs):
        '''
        '''
        x_v, x_vp = inputs
        return self.h_theta(tf.concat([x_v, x_vp-x_v,edge_attributes], -1))
    
    
    def build(self):
        pass

    
    def call(self, Adjacency, node_features, edge_attributes):
        inputs = (Adjacency, node_features, edge_attributes)
        if len(Adjacency.shape) == 2:
            return self._convolution(inputs)
        return tf.vectorized_map(self._convolution, inputs)

if __name__ == "__main__":
    # Test MLP class
    h_dim = 4
    o_dim = 8
    mlp = MLP(h_dim, o_dim)
    x = tf.random.uniform((10, 80))
    # Test EdgeConv (Batch)
    ec = EdgeConv(mlp)
    A = tf.convert_to_tensor([[[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]], 
                              [[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]]], dtype=tf.float32)
    A = tf.sparse.from_dense(A)
    Feat = tf.convert_to_tensor([[[1, 2, 6],[2, 3, 2], [4, 5, 10], [5, 6, -1]],
                                 [[1, 2, -3],[2, 3, 0.5], [4, 5, -9], [5, 6, 0]]], dtype=tf.float32)
    '''
    A = tf.convert_to_tensor([[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]])
    Feat = tf.convert_to_tensor([[1, 2],[2, 3], [4, 5], [5, 6]])
    '''
    print("test 2")
    print(A.shape)
    print(Feat.shape)
    print(ec(A,Feat).shape)