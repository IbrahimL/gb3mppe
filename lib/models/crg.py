import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from .layers import EdgeConvE, EdgeConv
from .layers import MLP


class CRG(Model):
    '''
    '''
    def __init__(self, mlp_dim, hidden_dim, output_dim):
        super(CRG, self).__init__()
        self.linear_hid1    = layers.Dense(hidden_dim[0], activation='relu')
        self.linear_hid2    = layers.Dense(hidden_dim[1], activation='relu')
        self.concatenate_layer = layers.Concatenate(axis=-1)
        self.EdgeConv_hid3 = EdgeConv(MLP(mlp_dim[0], mlp_dim[0]))
        self.EdgeConv_hid4 = EdgeConv(MLP(mlp_dim[1], mlp_dim[1]))
        self.EdgeConv_hid5 = EdgeConv(MLP(mlp_dim[2], mlp_dim[2]))
        self.linear_hid6   = layers.Dense(hidden_dim[2], activation='relu')
        self.linear_out    = layers.Dense(output_dim, activation='relu')
    
    def call(self, coordinates, adjacency, node_features, center_scores):
        x = self.linear_hid1(coordinates)
        x = self.linear_hid2(x)
        # A vérifier 
        x = layers.concatenate([x,node_features,center_scores])
        x = self.EdgeConv_hid3(adjacency, x)
        EdgeConv1_ouput = x
        x = self.EdgeConv_hid4(adjacency, x)
        EdgeConv2_ouput = x
        x=add([x,EdgeConv1_ouput])
        x = self.EdgeConv_hid5(adjacency, x)
        x=add([x,EdgeConv2_ouput])        
        x = self.linear_hid6(x)
        x = self.linear_out(x)
        return x

    @tf.function
    def train_step(data, labels):
        '''
        '''
        pass

    def train_loop(EPOCHS,train_ds):
        '''
        '''
        pass
