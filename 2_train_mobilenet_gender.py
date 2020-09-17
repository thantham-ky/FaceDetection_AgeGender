from keras.applications import Xception, MobileNetV2
from keras.models import  Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

#from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.optimizers import RMSprop, Adam

from keras.utils import plot_model
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

image_data_file = "imgdata.npy"
age_data_file = "age.npy"
gender_data_file = "gender.npy"

gender_history_file = "gender_history.csv"
gender_model_file = "gender_model.h5"

epoch = 20
validation_split = 0.2
random_seed = 7
n_processor = -1

print("[PROCESS] : load data")
image_data = np.load(image_data_file) / 255.
print("[INFO] : image data file - ", image_data_file)
gender_data = np.load(gender_data_file)
print("[INFO] : gender label file - ", gender_data_file)

print("[INFO] : image data dimension - ", image_data.shape[1:])
print("[INFO] : number of images - ", image_data.shape[0])

print("[PROCESS] : train and test set splitting")
x_train, x_test, y_train, y_test = train_test_split(image_data, gender_data, test_size=validation_split, random_state=random_seed)

print("[INFO] : number of training data - ", x_train.shape[0])
print("[INFO] : number of validation data - ", x_test.shape[0])

print("[PROCESS] : define image data generator")
datagen = ImageDataGenerator(horizontal_flip=True, 
                             rotation_range=10, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             zoom_range=0.1, 
                             rescale=0.1,
                             shear_range=0.1,
                             brightness_range=[0.1,0.2])

print("[PROCESS]: construct model")
base_model = Xception(input_shape=x_train.shape[1:], include_top=False, pooling='avg', weights='imagenet')
x = base_model.output
x = Dense(2048, activation='relu')(x)
x = Dropout(0.5)(x)
gender_out = Dense(1, activation='sigmoid', name='gender_out')(x)
gender_model = Model(inputs=base_model.input, outputs=gender_out)

print("[PROCESS] : model compile")
gender_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

print("[PROCESS] : model fitting")
gender_history = gender_model.fit_generator(datagen.flow(x_train, y_train),
                                      steps_per_epoch=x_train.shape[0]//32,epochs=epoch,
                                      validation_data=datagen.flow(x_test, y_test),
                                      workers=n_processor)

print("[PROCESS] : save history - ", gender_history_file)
gender_hist_df = pd.DataFrame(gender_history.history)
gender_hist_df.to_csv(gender_history_file)

print("[INFO] : save model - ", gender_model_file)
gender_model.save(gender_model_file)