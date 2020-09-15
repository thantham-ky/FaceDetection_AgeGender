from keras.applications import Xception, MobileNetV2
from keras.models import  Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

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

print("load data......", end="\n\n\n")
image_data = np.load(image_data_file)
gender_data = np.load(gender_data_file)
age_data = np.load(age_data_file)


print("training data shape: ", image_data.shape[1:])
print("number of training: ", image_data.shape[0])

print("train and test set splitting..... ", end="\n\n\n")

y = np.array([gender_data, age_data]).transpose()

x_train, x_test, y_train, y_test = train_test_split(image_data, y, test_size=0.1, random_state=0)

datagen = ImageDataGenerator(horizontal_flip=True, 
                             rotation_range=10, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             zoom_range=0.1, 
                             rescale=0.1)


print("Model constructing.....", end="\n\n\n")
base_model = MobileNetV2(input_shape=x_train.shape[1:], include_top=False, pooling='avg', weights='imagenet')
x = base_model.output
x = Dense(4096, activation='relu')(x)

gend_branch = Dropout(0.5)(x)
gender_out = Dense(1, activation='sigmoid', name='gender_out')(gend_branch)
gender_model = Model(inputs=base_model.input, outputs=gender_out)

gender_model.compile(loss={'gender_out': 'binary_crossentropy'}, optimizer=Adam(learning_rate=0.0001), metrics={'gender_out':'accuracy'})

gender_history = gender_model.fit_generator(datagen.flow(x_train, y_train[:,0]),
                                      steps_per_epoch=x_train.shape[0]//32,epochs=100,
                                      validation_data=datagen.flow(x_test, y_test[:,0]),
                                      workers=2,use_multiprocessing=True)

gender_hist_df = pd.DataFrame(gender_history.history)
gender_hist_df.to_csv(gender_history_file)

gender_model.save(gender_model_file)