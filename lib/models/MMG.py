import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import sys
from .layers import EdgeConvE, MLP


class MMG(Model):
    '''
    Implementation of the Multi-view Matching Graph Module using tensorflow (https://arxiv.org/pdf/2109.05885.pdf)
    '''
    def __init__(self, hidden_dim, output_dim):
        """[init method for the Multi-view Matching Graph Module]

        Args:
            hidden_dim (list of 3 integers [int]): [dimension of the MLP in the edge_conv layers]
            output_dim ([int]): [dimension of the output]
        """
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
        """[concatenate the features to calculate edge connectivity]

        Args:
            features (tf tensor of shape(B,V,C)): [the output features from the second edgeconv layer,
                                                   this is used to formulate edge connectivity problem,
                                                   B: Batch Size,
                                                   V: Number of nodes,
                                                   C: Number of features]

        Returns:
            feat (tf tensor of shape (B,V,2C)): [the concatenation of the features from each node with
                                                 with the features of all other nodes]
        """
        B, V, C = features.shape
        feat_source = tf.reshape(tf.repeat(features, V, axis=0), shape=[B, V*V, C])
        feat_target = tf.reshape(tf.repeat(features, V, axis=1), shape=[B, V*V, C])
        feat = tf.concat([feat_source, feat_target], axis=-1)
        return feat
    
    def call(self, adjacency, node_features, edge_attributes):
        """[forward pass of the MMG]

        Args:
            adjacency (tf sparse tensor of shape (B,V,V)): [adjacency matrices for every frame]
            node_features (tf tensor of shape (B,V,C1)): [feature vector for each node]
            edge_attributes (tf tensor of shape (B,V,V,C2)): [feature vector for each edge]

        Returns:
            output tf tensor of shape (B,V,V): [predicted adjacency matrices]
        """
        x = self.EdgeConvE_hid1(adjacency, node_features, edge_attributes)
        x = self.EdgeConvE_hid2(adjacency, x, edge_attributes)
        x = self._feat_concat(x)
        x = self.linear_hid3(x)
        x = self.linear_out(x)
        # je ne peux pas récuperer adjacency.shape !! 
        return tf.reshape(x,(edge_attributes.shape[0:3]))
    

    @tf.function
    def train_step(self, data, connectivity_target):
        """[preform a training step]

        Args:
            data: list [adjacency, node_features, edge_attributes] as defined in the call function
            connectivity_target: list of ground truth graphs

        Returns:
            loss: the loss value 
        """
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
        return loss

