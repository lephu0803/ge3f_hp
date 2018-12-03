#
# Genie
# Tensorflow Model Training
# Date : 08/05/2018
# Copyright: Ge3f Pte Ltd
#


import tensorflow as tf

import keras.backend as K
import keras
import numpy as np
import warnings
import os
# NOTE: remove 'tensorflow.python'
from inception_v3 import InceptionV3
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import glob
# from ipi_lib.inception_v3 import *
# from ipi_lib.plots import plot_model_history


# Only when facing GPU memory issues.

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
K.tensorflow_backend.set_session(tf.Session(config=config))


# def lr_scheduler(epoch):
#     if epoch < 20:
#         return 0.1
#     if epoch < 50:
#         return 0.01
#     return 0.001


# dirs need subdirs as classified datas
data_dir = '/home/ge3f/Documents/GE3F/HP_project/ColorDb/'
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
model_dir = '/home/ge3f/Documents/GE3F/HP_project/ColorDb/model'
model_weight_path = os.path.join(
    model_dir, "weights-improvement-{epoch:02d}-{val_acc:.4f}-{val_loss:f}.hdf5")
print(train_dir)
print(test_dir)
print(model_weight_path)


classes = ['BPAD','BPOX','BPOX0','DLAM','PART','PART0','underetch NOT reworkable','underetch reworkable']

img_size = 128
channel = 1
num_outputs = 3  # NOTE: unused var
batch_size = 4
epochs = 50
class_mode = 'categorical'
color_mode = 'grayscale'


# image_input = Input(shape=(img_size, img_size, channel))  # NOTE: unused var


# def subtract_mean_from_image(x):
#     # New_image = Image - dataset mean
#     mean_value = 94.216579534425989
#     x = x - mean_value1

#     return x


# NOTE: cannot call pre_processing_function -> deleted
datagen_train = ImageDataGenerator(
    rescale=1./255)
datagen_test = ImageDataGenerator(
    rescale=1./255)

generator_train = datagen_train.flow_from_directory(directory=train_dir,
                                                    batch_size=batch_size,
                                                    target_size=(
                                                        img_size, img_size),
                                                    shuffle=True,
                                                    class_mode=class_mode,
                                                    color_mode=color_mode)

generator_test = datagen_test.flow_from_directory(directory=test_dir,
                                                  batch_size=batch_size,
                                                  target_size=(
                                                      img_size, img_size),
                                                  color_mode=color_mode,
                                                  class_mode=class_mode,
                                                  shuffle=False)


y_train = generator_train.classes
print("This is what I am looking for {}".format(generator_test.n))

class_weight = compute_class_weight('balanced', np.unique(y_train), y_train)
print(class_weight)

pre_computed_weights = dict(zip(classes, class_weight))
print("Pre computed weights for each class : \n", pre_computed_weights)

steps_test = generator_test.n // batch_size
print("The steps for batch size {} of test set is {}".format(batch_size, steps_test))

steps_per_epoch = generator_train.n // batch_size
print("The steps for batch size {} of training set is {}".format(
    steps_per_epoch, steps_per_epoch))
print(type(generator_test.n))  # number of samples
model = InceptionV3(include_top=True)
model.summary()

pretrained_path = glob.glob('/home/ge3f/Documents/GE3F/HP_project/ColorDb/model/*')
if pretrained_path is not None:
    lastest_model = max(pretrained_path, key=os.path.getctime)
    print(lastest_model)
    model.load_weights(lastest_model)

opt = keras.optimizers.Adam(lr=0.001 )
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

check_point = keras.callbacks.ModelCheckpoint(
    model_weight_path, monitor='val_acc', save_best_only=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.3, patience=10)

callback_list = [check_point, reduce_lr]


model = model.fit_generator(generator_train,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=generator_test,
                            callbacks=callback_list,
                            validation_steps=steps_test,
                            class_weight=class_weight)


# plot_model_history(model)
