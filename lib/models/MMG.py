import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import sys
from .layers import EdgeConvE, MLP


class MMG(Model):
    '''
    '''
    def __init__(self, hidden_dim, output_dim):
        super(MMG, self).__init__()
        # pour le training 
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.adam = tf.keras.optimizers.Adam(learning_rate=1e-4,name='Adam')
        #
        self.EdgeConvE_hid1 = EdgeConvE(MLP(hidden_dim[0], hidden_dim[0]))
        self.EdgeConvE_hid2 = EdgeConvE(MLP(hidden_dim[1], hidden_dim[0]))
        self.linear_hid3    = layers.Dense(hidden_dim[2], activation='relu')
        self.linear_out     = layers.Dense(output_dim, activation='sigmoid')
    
    def _feat_concat(self, features):
        B, V, C = features.shape
        feat_source = tf.reshape(tf.repeat(features, V, axis=0), shape=[B, V*V, C])
        feat_target = tf.reshape(tf.repeat(features, V, axis=1), shape=[B, V*V, C])
        feat = tf.concat([feat_source, feat_target], axis=-1)
        return feat
    
    def call(self, adjacency, node_features, edge_attributes):
        x = self.EdgeConvE_hid1(adjacency, node_features, edge_attributes)
        x = self.EdgeConvE_hid2(adjacency, x, edge_attributes)
        x = self._feat_concat(x)
        x = self.linear_hid3(x)
        x = self.linear_out(x)
        # je ne peux pas récuperer adjacency.shape !! 
        return tf.reshape(x,(edge_attributes.shape[0:3]))
    

    @tf.function
    def train_step(self, data, connectivity_target):
        '''
        '''
        adjacency, node_features, edge_attributes = data[0],data[1],data[2]
        with tf.GradientTape() as tape:
            # forward pass
            predictions = self.call(adjacency, node_features,
                                    edge_attributes)
            loss = self.loss_object(connectivity_target,predictions)
        # calcul des gradients
        gradient = tape.gradient(loss, self.trainable_variables)
        # retropropagation
        self.adam.apply_gradients(zip(gradient, self.trainable_variables))

    @tf.function
    def train_loop(self,train_ds,EPOCHS=4):
        '''
        '''
        for epoch in range(EPOCHS):
            for data, connect_target in train_ds:
                self.train_step(data, connect_target)

