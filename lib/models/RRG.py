import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from .layers import EdgeConvE, EdgeConv, GlobalMaxPool
from .layers import MLP


class RRG(Model):
    '''
    '''
    def __init__(self, mlp_dim, hidden_dim, output_dim):
        super(RRG, self).__init__()
        self.linear_hid1    = layers.Dense(hidden_dim[0], activation='relu')
        self.linear_hid2    = layers.Dense(hidden_dim[1], activation='relu')
        self.concatenate_layer = layers.Concatenate(axis=-1)
        self.EdgeConvE_hid1 = EdgeConvE(MLP(mlp_dim[0], mlp_dim[0]))
        self.EdgeConvE_hid2 = EdgeConvE(MLP(mlp_dim[1], mlp_dim[1]))
 #       self.MaxPool_hid1   = layers.MaxPool2D(mlp_dim[1])
        self.MaxPool_hid1   = GlobalMaxPool()
        self.linear_hid3    = layers.Dense(hidden_dim[2], activation='relu')
        self.EdgeConv_hid1  = EdgeConv(MLP(mlp_dim[2], mlp_dim[2]))
        self.EdgeConv_hid2  = EdgeConv(MLP(mlp_dim[3], mlp_dim[3]))
        self.EdgeConv_hid3  = EdgeConv(MLP(mlp_dim[4], mlp_dim[4]))
        self.linear_hid4    = layers.Dense(hidden_dim[3], activation='relu')
        self.linear_out1    = layers.Dense(output_dim[0], activation='relu')
        self.linear_hid5    = layers.Dense(hidden_dim[4], activation='relu')
        self.linear_out2    = layers.Dense(output_dim[1], activation='relu')
    def call(self, coordinates, adjacency, node_features, edge_features, joint_types):
        x1 = self.linear_hid1(coordinates)
        x1 = self.linear_hid2(x1)
        x1 = self.concatenate_layer([x1, node_features, joint_types])
        x1 = self.EdgeConvE_hid1(adjacency, x1, edge_features)
        x1 = self.EdgeConvE_hid2(adjacency, x1, edge_features)
        x1 = self.MaxPool_hid1(x1)
        x1 = self.linear_hid3(x1)
        x1 = self.EdgeConv_hid1(adjacency, x1)
        EdgeConv1_ouput = x1
        x1 = self.EdgeConv_hid2(adjacency, x1)
        EdgeConv2_ouput = x1
        x1 = add([x1,EdgeConv1_ouput])
        x1 = self.EdgeConv_hid3(x1)
        x1 = add([x1,EdgeConv2_ouput])
        x2 = x1
        x1 = self.linear_out1(x1) 
        x1 = self.linear_out1(x1)   
        x2 = self.linear_out2(x2)  
        x2 = self.linear_out2(x2)
        return x1,x2

    @tf.function
    def train_step(data, labels):
        '''
        '''
        pass

    def train_loop(EPOCHS,train_ds):
        '''
        '''
        pass
