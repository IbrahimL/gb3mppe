import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import add
from layers_def import EdgeConvE
from layers_def import MLP


class CRG(Model):
    '''
    '''
    def __init__(self,mlp_dim,hidden_dim,output_dim):
        super(CRG, self).__init__()
        self.linear_hid1    = layers.Dense(hidden_dim, activation='relu')
        self.linear_hid2    = layers.Dense(hidden_dim, activation='relu')
        self.EdgeConv_hid3 = EdgeConv (MLP(mlp_dim[0]))
        self.EdgeConv_hid4 = EdgeConv (MLP(mlp_dim[1]))
        self.EdgeConv_hid5 = EdgeConv (MLP(mlp_dim[1]))
        self.linear_hid6   = layers.Dense(hidden_dim, activation='relu')
        self.linear_out    = layers.Dense(output_dim, activation='relu')
    
    def call(self, input):
        [Coordinate3D,Feature512D,CenterScore]=input
        x = self.linear_hid1(Coordinate3D)
        x = self.linear_hid2(x)
        # A vérifier 
        x = layers.concatenate()[x,Feature512D,CenterScore]
        x = self.EdgeConv_hid3(x)
        EdgeConv1_ouput = x
        x = self.EdgeConv_hid4(x)
        EdgeConv2_ouput = x
        x=add([x,EdgeConv1_ouput])
        x = self.EdgeConv_hid5(x)
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
