from functools import reduce
from typing import Any, Callable, Dict
import tensorflow_graphics
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow_graphics.geometry.convolution.graph_convolution import edge_convolution_template
from tensorflow_graphics.geometry.convolution import utils
from tensorflow.python.ops.gen_math_ops import Add

#https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/convolution/graph_convolution.py   
def edge_convolution_E_template(data, neighbors, edges_features, sizes, edge_function, reduction, edge_function_kwargs, name="Layer_E"):
  with tf.name_scope(name):
    data = tf.convert_to_tensor(value=data)
    edges_features = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=edges_features)
    neighbors = tf.compat.v1.convert_to_tensor_or_sparse_tensor(value=neighbors)
    B, V, _, C = edges_features.shape
    if sizes is not None:
      sizes = tf.convert_to_tensor(value=sizes)

    data_ndims = data.shape.ndims
    utils.check_valid_graph_convolution_input(data, neighbors, sizes)

    # Flatten the batch dimensions and remove any vertex padding.
    if data_ndims > 2:
      if sizes is not None:
        sizes_square = tf.stack((sizes, sizes), axis=-1)
      else:
        sizes_square = None
      x_flat, unflatten = utils.flatten_batch_to_2d(data, sizes)
      adjacency = utils.convert_to_block_diag_2d(neighbors, sizes_square)
    else:
      x_flat = data
      adjacency = neighbors
    adjacency_ind_0 = adjacency.indices[:, 0]
    adjacency_ind_1 = adjacency.indices[:, 1]
    vertex_features = tf.gather(x_flat, adjacency_ind_0)
    neighbor_features = tf.gather(x_flat, adjacency_ind_1)
    edge_att_ind_0 = adjacency_ind_0 // V
    edge_att_ind_1 = adjacency_ind_0 % V
    edge_att_ind_2 = adjacency_ind_1 % V
    edges_indices = tf.convert_to_tensor([edge_att_ind_0, edge_att_ind_1, edge_att_ind_2], dtype=tf.int32)
    edges_indices = tf.transpose(edges_indices, perm=[1,0])
    edge_features = tf.gather_nd(edges_features, edges_indices)
    output = edge_function(vertex_features, neighbor_features, edge_features,
                                  **edge_function_kwargs)
    if reduction == "weighted":
      output_weighted = output * tf.expand_dims(
          adjacency.values, -1)
      features = tf.math.unsorted_segment_sum(
          data=output_weighted,
          segment_ids=adjacency_ind_0,
          num_segments=tf.shape(input=x_flat)[0])
    elif reduction == "max":
      features = tf.math.segment_max(
          data=output, segment_ids=adjacency_ind_0)
    else:
      raise ValueError("The reduction method must be 'weighted' or 'max'")

    features.set_shape(
        features.shape.merge_with(
            (tf.compat.dimension_value(x_flat.shape[0]),
             tf.compat.dimension_value(output.shape[-1]))))

    if data_ndims > 2:
      features = unflatten(features)
    return features

class MLP(Model):
    '''
    '''
    def __init__(self, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_hid1 = layers.Dense(hidden_dim, activation='relu')
        self.linear_hid2 = layers.Dense(hidden_dim, activation='relu')
        self.linear_out = layers.Dense(output_dim, activation='relu')
    
    def call(self, input):
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
    def __init__(self, h_theta):
        super(EdgeConvE, self).__init__()
        self.h_theta = h_theta
        
    def _aggregate(self, x_v, x_vp, e_v_vp):
        '''
        '''
        return self.h_theta(tf.concat([x_v, x_vp-x_v, e_v_vp], -1))
    
    def call(self, Adjacency, node_features, edge_attributes):
        return edge_convolution_E_template(data=node_features, 
                                         neighbors=Adjacency,
                                         edges_features=edge_attributes, 
                                         sizes=None,
                                         edge_function=self._aggregate,
                                         reduction='max',
                                         edge_function_kwargs={},
                                         name='edge_conv')

if __name__ == "__main__":
    import numpy as np
    # Test MLP class
    h_dim = 4
    o_dim = 8
    mlp_1 = MLP(h_dim, o_dim)
    x = tf.random.uniform((10, 80))
    # Test EdgeConv (Batch)
    ec = EdgeConv(mlp_1)
    A = tf.convert_to_tensor([[[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]], 
                              [[1, 0, 1, 0],[0, 1, 1, 1],[1, 1, 1, 0],[0, 1, 0, 1]]], dtype=tf.float32)
    A = tf.sparse.from_dense(A)
    edge_att = tf.convert_to_tensor([[[[1,1], [0,0], [1,1], [0,0]],[[0,0], [1,1], [1,1], [1,1]],[[1,1], [1,1], [1,1], [0,0]],[[0,0], [1,1], [0,0], [1,1]]], 
                              [[[1,1], [0,0], [1,1], [0,0]],[[0,0], [1,1], [1,1], [1,1]],[[1,1], [1,1], [1,1], [0,0]],[[0,0], [1,1], [0,0], [1,1]]]], dtype=tf.float32)
    Feat = tf.convert_to_tensor([[[1, 2, 6],[2, 3, 2], [4, 5, 10], [5, 6, -1]],
                                 [[1, 2, -3],[2, 3, 0.5], [4, 5, -9], [5, 6, 0]]], dtype=tf.float32)
    
    print("test Edge Convolution")
    print('adjacency:', A.shape)
    print('node features:', Feat.shape)
    print('output:', ec(A,Feat).shape)
    ### TEST EDGE CONV E
    mlp_2 = MLP(h_dim, o_dim)
    ece = EdgeConvE(mlp_2)
    print("test Edge Convolution-E")
    print('adjacency:', A.shape)
    print('node features:', Feat.shape)
    print('edge attributes:', edge_att.shape)
    print('output:', ece(A,Feat,edge_att).shape)
    
