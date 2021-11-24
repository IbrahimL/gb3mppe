import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from layers_def import EdgeConvE
from layers_def import MLP


class RRG(Model):
    '''
    '''
    def __init__(self,mlp_dim,hidden_dim,output_dim1,output_dim2):
        super(RRG, self).__init__()
        self.linear_hid1    = layers.Dense(hidden_dim, activation='relu')
        self.linear_hid2    = layers.Dense(hidden_dim, activation='relu')
        self.EdgeConvE_hid1 = EdgeConvE (MLP(mlp_dim[0]))
        self.EdgeConvE_hid1 = EdgeConvE (MLP(mlp_dim[1]))
        # a verifier
        self.MaxPool_hid1   = layers.MaxPool2D (hidden_dim)
        self.linear_hid3    = layers.Dense(hidden_dim, activation='relu')
        self.EdgeConv_hid1  = EdgeConv (MLP(mlp_dim[0]))
        self.EdgeConv_hid2  = EdgeConv (MLP(mlp_dim[0]))
        self.EdgeConv_hid3  = EdgeConv (MLP(mlp_dim[0]))
        self.linear_hid4    = layers.Dense(hidden_dim, activation='relu')
        self.linear_out1    = layers.Dense(output_dim1, activation='relu')
        self.linear_hid5    = layers.Dense(hidden_dim, activation='relu')
        self.linear_out2    = layers.Dense(output_dim2, activation='relu')
    
    def call(self, input):
        [Coordinate3D,Feature512D,JointType]=input
        x = self.linear_hid1(Coordinate3D)
        x = self.linear_hid2(x)
        x = layers.concatenate()[x,Feature512D,JointType]
        x = self.EdgeConvE_hid1(x)
        x = self.EdgeConvE_hid1(x)
        x = self.MaxPool_hid1 (x)
        x = self.linear_hid3 (x)
        x = self.EdgeConv_hid1 (x)
        EdgeConv1_ouput = x
        x = self.EdgeConv_hid2 (x)
        EdgeConv2_ouput = x
        x = add([x,EdgeConv1_ouput])
        x = self.EdgeConv_hid3 (x)
        x = add([x,EdgeConv2_ouput])
        x2 = x
        x = self.linear_hid4  (x) 
        x = self.linear_out1  (x)   
        x2 = self.linear_hid5 (x2)  
        x2 = self.linear_out2 (x2)
        return x,x2

    @tf.function
    def train_step(data, labels):
        '''
        '''
        pass

    def train_loop(EPOCHS,train_ds):
        '''
        '''
        pass
