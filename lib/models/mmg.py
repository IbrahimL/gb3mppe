import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from layers import EdgeConvE
from layers import MLP


class MMG(Model):
    '''
    '''
    def __init__(self):
        super(MMG, self).__init__()
        # pour le training 
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.adam = tf.keras.optimizers.Adam(learning_rate=1e-4,name='Adam')
        # archi
        h_theta1 = MLP(512, 256)
        self.EdgeConvE_hid1 = EdgeConvE (h_theta1)
        h_theta2 = MLP(256, 128)        
        self.EdgeConvE_hid2 = EdgeConvE (h_theta2)
        self.linear_hid3    = layers.Dense(64, activation='relu')
        self.linear_out     = layers.Dense(1, activation='sigmoid')
    
    def call(self, input):
        # A enlever  ! 
#        if len(input.shape) == 1:
 #           input = tf.expand_dims(input, axis=0)
        x = self.EdgeConvE_hid1(input)
        x = self.EdgeConvE_hid2(x)
        x = self.linear_hid3(x)
        x = self.linear_out(x)
        return x
    

    @tf.function
    def train_step(data, labels):
        '''
        '''
        with tf.GradientTape() as tape:
            # forward pass
            predictions = MMG(data)
            loss = self.loss_object(labels, predictions)
        # calcul des gradients
        gradient = tape.gradient(loss, MMG.trainable_variables)
        # retropropagation
        self.adam.apply_gradients(zip(gradient, MMG.trainable_variables))
    @tf.function
    def train_loop(self,train_ds,EPOCHS=2):
        '''
        '''
        for epoch in range(EPOCHS):
            for data, labels in train_ds:
                self.train_step(data, labels)

