import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import confusion_matrix, accuracy_score
import datetime
from conv_models import efb0_ft_gap
from utils import plot_accuracy, plot_confusion_matrix, plot_loss, prepare

'''
Galadrial: 
https://forums.developer.nvidia.com/t/could-not-create-cudnn-handle-cudnn-status-alloc-failed/108261/3
https://www.tensorflow.org/guide/gpu
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
Check GPU's are available or not
'''
print("GPU: ", tf.config.list_physical_devices('GPU'))

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':

    ''' Filenames '''
    weight_filepath = 'models/workspace/cv/weights/dwtfruitnet/effnb0_iter1/ft_effnetb0_e20.h5'
    fintetune_acc_filename = 'models/workspace/cv/weights/dwtfruitnet/effnb0_iter1/ft_effnetb0_e20_acc_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    finetune_loss_filename = 'models/workspace/cv/weights/dwtfruitnet/effnb0_iter1/ft_effnetb0_e20_loss_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    cm_filename = 'models/workspace/cv/weights/dwtfruitnet/effnb0_iter1/ft_effnetb0_e20_cm_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logfile = 'models/workspace/cv/weights/dwtfruitnet/effnb0_iter1/ft_effnetb0_e20_log_{}.csv'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir = "models/workspace/cv/logs/ft_effnetb0_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    plot_model_filename = "models/workspace/cv/weights/dwtfruitnet/ft_effnetb0_model.png"
    test_acc_filename = "models/workspace/cv/weights/dwtfruitnet/effnb0_iter1/ft_effnetb0_e20.txt"
    ''' Tensorboard '''
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    ''' Dataset '''
    train_val_dirname = 'models/workspace/cv/data/imgcls/UOSFruitNet/train'
    test_dirname = 'models/workspace/cv/data/imgcls/UOSFruitNet/test'
    nb_classes = 12
    batch_size = 16
    csv_logger = CSVLogger(logfile, append=True, separator=';')
    '''
    https://kvirajdatt.medium.com/starting-with-tensorflow-datasets-part-2-intro-to-tfds-and-its-methods-32d3ac36420f
    https://stackoverflow.com/questions/73678683/loading-data-using-tensorflow-datasets-and-splitting
    Problem:: ValueError: Shapes (None, 1) and (None, 9) are incompatible
    Solution:: label_mode = 'categorical' means that the labels are encoded as a categorical vector (e.g. for categorical_crossentropy loss)
    '''
    train_ds, pre_val_ds = tf.keras.utils.image_dataset_from_directory(directory=train_val_dirname,seed=42,
                                                                   subset ='both', image_size=(224,224), shuffle=True,
                                                                   batch_size=batch_size, validation_split=0.30,
                                                                   label_mode = 'categorical', interpolation="bilinear"
                                                                )
    len_pre_val_ds = pre_val_ds.cardinality().numpy() #len(pre_val_ds)
    test_size = int(0.5 * len_pre_val_ds)
    test_ds = pre_val_ds.take(test_size)
    val_ds = pre_val_ds.skip(test_size)

    class_names = train_ds.class_names 
    print("class_names: ", len(class_names), " , ", class_names)
    print("len(train_ds, val_ds, test_ds): ", len(train_ds), " , ", len(val_ds), " , ", len(test_ds))
    print("type(train_ds, val_ds, test_ds):\n ", type(train_ds), " \n ", type(val_ds), " \n ", type(test_ds))

    data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal_and_vertical"),
                                            layers.RandomRotation(0.5)
                                            ])
    
    train_ds = prepare(train_ds, data_augmentation, shuffle=False, augment=True)
    val_ds = prepare(val_ds, data_augmentation)
    test_ds = prepare(test_ds, data_augmentation)
  
    model_save = ModelCheckpoint(weight_filepath, 
                             save_best_only = True, 
                             save_weights_only = True,
                             monitor = 'val_loss', 
                             mode = 'min', verbose = 1)
    
    early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, 
                           patience = 5, mode = 'min', verbose = 1,
                           restore_best_weights = True)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', 
                                                patience=2, 
                                                verbose=1, 
                                                factor=0.3, 
                                                min_delta = 0.001, 
                                                mode = 'min'
                                            )
    model = efb0_ft_gap(nb_classes=nb_classes)
       
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate= 0.001, momentum = 0.9)
    model.compile(optimizer= optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                )
    #print(model.summary())
    plot_model(model, to_file=plot_model_filename, show_shapes=True)
    
    history = model.fit(train_ds,
                        validation_data = val_ds,
                        epochs = 20,
                        verbose = 1,
                        callbacks = [model_save, early_stop, reduce_lr, csv_logger, tensorboard_callback]
                    )
    
    plot_accuracy(history, acc_filename= fintetune_acc_filename) 
    plot_loss( history, loss_filename = finetune_loss_filename)
    print("Model training done !")
    
    # Inference 
    #all_test_ds = test_ds.unbatch()
    y_pred = []  # store predicted labels
    y_true = []  # store true labels

    for image_batch, label_batch in test_ds:
      print(tf.shape(image_batch))
      # append true labels
      y_true.append(label_batch)
      # compute predictions
      preds = model.predict(image_batch, verbose=0)
      # append predicted labels
      y_pred.append(np.argmax(preds, axis = - 1))
    
    # convert the true and predicted labels into tensors
    correct_labels = tf.concat([item for item in y_true], axis = 0)
    rounded_correct_labels=np.argmax(correct_labels, axis=1)
    predicted_labels = tf.concat([item for item in y_pred], axis = 0)
    print("Correct and Predicted Lables:\n", rounded_correct_labels ,"\n", predicted_labels)
    print("Testing Accuarcy : ", accuracy_score(rounded_correct_labels, predicted_labels))

    with open(test_acc_filename, "a") as f:
       print("Correct and Predicted Lables:\n", rounded_correct_labels ,"\n", predicted_labels, file=f)
       print("Testing Accuarcy : ", accuracy_score(rounded_correct_labels, predicted_labels), file=f)

    cm = confusion_matrix(rounded_correct_labels, predicted_labels)
    plot_confusion_matrix(cm,class_names, cm_filename)
    tf.keras.backend.clear_session()
    print("Finished !")