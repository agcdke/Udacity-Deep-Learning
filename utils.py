import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import numpy as np
from wavetf import WaveTFFactory
from tensorflow.keras.layers import Conv2D, Concatenate

# https://stackoverflow.com/questions/54959929/concatenate-multiple-cnn-models-in-keras
# https://stackoverflow.com/questions/68997513/concatenate-two-layers-in-keras-tensorflow
# https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
# https://ai.stackexchange.com/questions/5769/in-a-cnn-does-each-new-filter-have-different-weights-for-each-input-channel-or
# https://chl260.github.io/PDF/Lee_PAMI17.pdf
# https://stackoverflow.com/questions/62320080/custom-convolutions-and-none-type-object-in-keras-custom-layer-for-gating-operat
'''
https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/ 
Example of 2D Convolutional Layer:
Again, we can constrain the input, in this case to a square 8×8 pixel input image with a single channel (e.g. grayscale) 
with a single vertical line in the middle.
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]
[0, 0, 0, 1, 1, 0, 0, 0]

The input to a Conv2D layer must be four-dimensional.

The first dimension defines the samples; in this case, there is only a single sample. 
The second dimension defines the number of rows; in this case, eight. 
The third dimension defines the number of columns, again eight in this case, 
and finally the number of channels, which is one in this case.

Therefore, the input must have the four-dimensional shape [samples, rows, columns, channels] or 
[1, 8, 8, 1] in this case.
'''

'''
Plot loss and accuracy as a function of the epoch, for the training and validation datasets.
Input param: training history, yrange value, accuracy filename, loss filename
Return: None (files are saved in directory).
'''
class Concat_WvHaarConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Concat_WvHaarConv, self).__init__(**kwargs)
        self.filter = 12
        self.kernel_size = 4
        self.haarconv = WaveTFFactory.build(kernel_type='haar')
        self.conv = Conv2D(self.filter, self.kernel_size, activation='relu', kernel_initializer='glorot_uniform')
       
    def call(self, x):
        x1 = self.haarconv(x)
        x2 = self.conv(x)
        output = Concatenate()([x1,x2])
        return output
    
class Concat_WvDbConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Concat_WvDbConv, self).__init__(**kwargs)
        self.filter = 12
        self.kernel_size = 4
        self.dbconv = WaveTFFactory.build(kernel_type='db2')
        self.conv = Conv2D(self.filter, self.kernel_size, activation='relu', kernel_initializer='glorot_uniform')

    def call(self, x):
        x1 = self.dbconv(x)
        x2 = self.conv(x)
        output = Concatenate()([x1,x2])
        return output
    
class Concat_WvHaarDb(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Concat_WvHaarDb, self).__init__(**kwargs)
        self.filter = 12
        self.kernel_size = 4
        self.haarconv = WaveTFFactory.build(kernel_type='haar')
        self.dbconv = WaveTFFactory.build(kernel_type='db2')
        self.conv = Conv2D(self.filter, self.kernel_size, activation='relu', kernel_initializer='glorot_uniform')

    def call(self, x):
        x1 = self.haarconv(x)
        x2 = self.dbconv(x)
        x3 = self.conv(x)
        output = Concatenate()([x1,x2,x3])
        return output
    
class Concat_WvHaarDbConv(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Concat_WvHaarDbConv, self).__init__(**kwargs)
        self.haarconv = WaveTFFactory.build(kernel_type='haar')
        self.dbconv = WaveTFFactory.build(kernel_type='db2')

    def call(self, x):
        x1 = self.haarconv(x)
        x2 = self.dbconv(x)
        output = Concatenate()([x1,x2])
        return output

class Gated_MaxAvgPooling(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Gated_MaxAvgPooling, self).__init__(**kwargs)
        '''
        tf.nn.conv2d : filters = A Tensor. Must have the same type as input. 
        A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
        '''
        self.mask = self.add_weight(name='mask', shape=(2,2,1,1),  # channel-wise: shape=(2,2,1,1), tf.keras.layers.Conv2D: shape=(2,2,3,3)
                                   initializer='glorot_uniform', trainable=True)
        self.maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding='VALID')
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='VALID')

    def call(self, x):

        self.batch, self.row, self.col, self.channel = x.shape
        self.output_size = self.row//2

        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        
        xs = []
        for c in tf.split(x, self.channel, 3):
            xs.append(tf.nn.conv2d(c, filters=self.mask, strides=[2,2], padding='VALID'))

        xs = tf.concat(xs, axis=-1)
        '''
        # for shape=(2,2,3,3) at name='mask'
        xs = tf.nn.conv2d(x, filters=self.mask, strides=[2,2], padding='VALID')
        '''
        z = tf.math.sigmoid(xs)
        output = tf.add(tf.multiply(z, x1), tf.multiply((1-z), x2))

        return output
    
def plot_accuracy(history, acc_filename):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    # Plot training and validation accuracy per epoch
    plt.plot(acc, color = 'purple', label='accuracy')
    plt.plot(val_acc, color = 'blue', label='val_accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    # Plot training and validation accuracy
    plt.savefig(fname=acc_filename)
    plt.figure().clear()

def plot_loss(history, loss_filename):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, color = 'green', label='loss')
    plt.plot(val_loss, color = 'red', label='val_loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    # Plot training and validation loss
    plt.savefig(fname=loss_filename)
    plt.figure().clear()

def plot_confusion_matrix(cm, classes, cm_filename,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname=cm_filename)
    plt.figure().clear()

'''
Prepare dataset for training, and val, test. Image augmentation on training dataset only.
Input param: dataset, resize_and_rescale and data_augmentation functions
Return: Augmented or Un-augmented dataset 
'''
def prepare(ds, data_augmentation, shuffle=False, augment=False, AUTOTUNE = tf.data.AUTOTUNE):

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets.
  #ds = ds.batch(batch_size)
  # Use data augmentation only on the training set.
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':
    
    X = np.random.uniform(0,1, (32,7,7,1280)).astype('float32')
    gp = Gated_MaxAvgPooling()
    print(gp(X).shape)
    gmap = Concat_WvHaarDbConv()
    print(gmap(X).shape)
    '''
    a = tf.constant([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
    b = tf.keras.activations.sigmoid(a)
    print(b.numpy())
    c = tf.math.sigmoid(a)
    print(c.numpy())
    '''