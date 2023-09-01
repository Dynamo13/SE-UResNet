from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

def conv_block(x, filter_size, size, dropout,num, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same",name="conv"+str(num))(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def se_block(x,r):
    copy=x
    gap=layers.GlobalAveragePooling2D()(x)
    flat=layers.Flatten()(gap)
    dense=layers.Dense(flat.shape[-1]//r, activation = 'relu')(gap)
    dense=layers.Dense(flat.shape[-1], activation = 'sigmoid')(dense)
    m =layers.multiply([dense,copy])
    return m

def resb(x, filter_size, size, dropout,num, batch_norm=False):
    # copy tensor to variable called x_skip
    x_skip = x
    x_skip=layers.Conv2D(1, (1, 1), padding="same")(x_skip)
    print(x_skip.shape)
    # Layer 1
    x = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    print(x.shape)
    # Layer 2
    x = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        x = layers.BatchNormalization(axis=3)(x)
    x = layers.Activation("relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    print(x.shape)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x