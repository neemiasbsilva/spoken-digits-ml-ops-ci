import numpy as np
import tensorflow as tf


class CNN_Architecture():

    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.input = tf.keras.layers.Input(shape=input_shape)


    def cnn_block(self, x, filters, kernel_size, strides=(1, 1), pool_size=(2,2)):
        x = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size,
            strides=strides, activation="relu", 
            kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(x)
        x = tf.keras.layers.SpatialDropout2D(rate=0.25, data_format="channels_last")(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)

        return x

    def get_model(self, filters, kernel_size, **kwargs):

        for i, (f, k) in enumerate(zip(filters, kernel_size)):
            if (i == 0):
                x = self.cnn_block(self.input, f, k)
            else:
                x = self.cnn_block(x, f, k)
        
        x = tf.keras.layers.GlobalAveragePooling2D(data_format="channels_last")(x)
        
        x = tf.keras.layers.Dense(512, activation="relu")(x)

        x = tf.keras.layers.Dense(self.n_classes, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=self.input, outputs=x)

        return model