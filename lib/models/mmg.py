import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from .layers import EdgeConvE, MLP


class MMG(Model):
    '''
    '''
    def __init__(self, hidden_dim, output_dim):
        super(MMG, self).__init__()
        self.EdgeConvE_hid1 = EdgeConvE(MLP(hidden_dim[0], hidden_dim[0]))
        self.EdgeConvE_hid2 = EdgeConvE(MLP(hidden_dim[1], hidden_dim[0]))
        self.linear_hid3    = layers.Dense(hidden_dim[2], activation='relu')
        self.linear_out     = layers.Dense(output_dim, activation='sigmoid')
    
    def call(self, input):
        x = self.EdgeConvE_hid1(input)
        x = self.EdgeConvE_hid2(x)
        x = self.linear_hid3(x)
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
