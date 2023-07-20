import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, AveragePooling2D,GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from wavetf import WaveTFFactory 
from utils import Concat_WvHaarDb

'''
https://towardsdatascience.com/transfer-learning-using-mobilenet-and-keras-c75daf7ff299
https://stackoverflow.com/questions/55571664/mobilenets-for-a-custom-image-size
'''

def efb0_ft_gap(nb_classes):
    base_model = tf.keras.applications.EfficientNetB0(
                                            input_shape = (224,224,3),
                                            include_top=False, weights="imagenet"
                                            )
    base_model.trainable = False
    x = base_model.output
    x= GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', name='fc1', kernel_initializer="glorot_uniform")(x)
    preds = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model

def efb0_ft_gmp(nb_classes):
    base_model = tf.keras.applications.EfficientNetB0(
                                            input_shape = (224,224,3),
                                            include_top=False, weights="imagenet"
                                            )
    base_model.trainable = False
    x = base_model.output
    x= GlobalMaxPooling2D()(x)
    x = Dense(512, activation='relu', name='fc1', kernel_initializer="glorot_uniform")(x)
    preds = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model


def efb0_ft_wvhaar(nb_classes):
    base_model = tf.keras.applications.EfficientNetB0(
                                            input_shape = (224,224,3),
                                            include_top=False, weights="imagenet"
                                            )
    base_model.trainable = False
    x = base_model.output
    x=WaveTFFactory.build(kernel_type='haar')(x)
    x=Flatten()(x)
    x = Dense(512, activation='relu', name='fc1', kernel_initializer="glorot_uniform")(x)
    preds = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model

def efb0_ft_wvdb(nb_classes):
    base_model = tf.keras.applications.EfficientNetB0(
                                            input_shape = (224,224,3),
                                            include_top=False, weights="imagenet"
                                            )
    base_model.trainable = False
    x = base_model.output
    x=WaveTFFactory.build(kernel_type='db2', dim=2)(x)
    x=Flatten()(x)
    x = Dense(512, activation='relu', name='fc1', kernel_initializer="glorot_uniform")(x)
    preds = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model


def efb0_ft_concat_wvhaardb(nb_classes):
    base_model = tf.keras.applications.EfficientNetB0(
                                            input_shape = (224,224,3),
                                            include_top=False, weights="imagenet"
                                            )
    base_model.trainable = False
    x = base_model.output
    x=Concat_WvHaarDb()(x)
    x=Flatten()(x)
    x = Dense(512, activation='relu', name='fc1', kernel_initializer="glorot_uniform")(x)
    preds = Dense(nb_classes, activation = 'softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)
    return model

if __name__ =='__main__':
    print(efb0_ft_concat_wvhaardb(12).summary())