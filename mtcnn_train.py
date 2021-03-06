from keras.applications import Xception, MobileNetV2
from keras.models import  Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from keras.optimizers import RMSprop, Adam

from keras.utils import plot_model
from keras.utils.vis_utils import pydot
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--imgdata' , dest='imgdata', default='imgdata.npy')
parser.add_argument('-a', '--age'  , dest='age', default='age.npy')
parser.add_argument('-g', '--gender'  , dest='gender', default='gender.npy')

parser.add_argument('-t', '--trend', dest='hist', default='history.csv')
parser.add_argument('-m', '--model'  , dest='model', default='final_model.h5')
parser.add_argument('-c', '--checkpoint'  , dest='checkpoint', default='model_checkpoint.h5')

args = parser.parse_args()


# image_data_file = "D:\\thantham\\FaceDetection_AgeGender\\imgdata.npy"
# age_data_file = "D:\\thantham\\FaceDetection_AgeGender\\age.npy"
# gender_data_file = "D:\\thantham\\FaceDetection_AgeGender\\gender.npy"

# history_file = "D:\\thantham\\FaceDetection_AgeGender\\history.csv"
# model_file = "D:\\thantham\\FaceDetection_AgeGender\\final_model.h5"
# model_checkpointfile = "D:\\thantham\\FaceDetection_AgeGender\\model_checkpoint.h5"

image_data_file = args.imgdata
age_data_file = args.age
gender_data_file = args.gender

history_file = args.hist
model_file = args.model
model_checkpointfile = args.checkpoint


print("load data......", end="\n\n\n")
image_data = np.load(image_data_file)
gender_data = np.load(gender_data_file)
age_data = np.load(age_data_file)


print("training data shape: ", image_data.shape[1:])
print("number of training: ", image_data.shape[0])

print("train and test set splitting..... ", end="\n\n\n")
#y = np.array([gender_data, age_data]).transpose()
y = np.array([gender_data, age_data]).transpose()
#y = pd.DataFrame([gender_data, age_data]).transpose()
x_train, x_test, y_train, y_test = train_test_split(image_data, y, test_size=0.1, random_state=0)

datagen = ImageDataGenerator(horizontal_flip=True, 
                             rotation_range=10, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1, 
                             zoom_range=0.1, 
                             rescale=0.1)


print("Model constructing.....", end="\n\n\n")

callbacks = [ #EarlyStopping(monitor='loss', patience=10, verbose=1), 
              ModelCheckpoint(model_checkpointfile, monitor='loss', save_best_only=True, verbose=1),
              ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]


base_model = MobileNetV2(input_shape=x_train.shape[1:], include_top=False, pooling='avg', weights='imagenet')
x = base_model.output
x = Dense(1024, activation='relu')(x)

#gend_branch = Dense(1024, activation='relu')(x)
gend_branch = Dropout(0.5)(x)
gender_out = Dense(1, activation='sigmoid', name='gender_out')(gend_branch)


#age_branch = Dense(1024, activation='relu')(x)
age_branch = Dropout(0.5)(x)
age_out = Dense(1, name='age_out')(age_branch)

age_model = Model(inputs=base_model.input, outputs=age_out)
gender_model = Model(inputs=base_model.input, outputs=gender_out)

age_model.compile(loss={'age_out': 'mse'}, optimizer=Adam(learning_rate=0.0001), metrics={'age_out':'mae'})
gender_model.compile(loss={'gender_out': 'binary_crossentropy'}, optimizer=Adam(learning_rate=0.0001), metrics={'gender_out':'accuracy'})


print("Fitting.....", end="\n\n\n")

age_history = age_model.fit_generator(datagen.flow(x_train, y_train[:,1]),
                                      steps_per_epoch=x_train.shape[0]//32,epochs=100,
                                      validation_data=datagen.flow(x_test, y_test[:,1]),
                                      workers=-1,use_multiprocessing=True)

gender_history = age_model.fit_generator(datagen.flow(x_train, y_train[:,0]),
                                      steps_per_epoch=x_train.shape[0]//32,epochs=100,
                                      validation_data=datagen.flow(x_test, y_test[:,0]),
                                      workers=-1,use_multiprocessing=True)
# history = model.fit_generator(datagen)

# history = model.fit(x_train, {'gender_out':y_train[:, 0],'age_out':y_train[:, 1]}, 
#                     validation_data= (x_test, {'gender_out':y_test[:, 0],'age_out':y_test[:, 1]}),
#                     batch_size=32, epochs=100, workers=4, callbacks=callbacks)

# age_hist_df = pd.DataFrame(history.history)
# age_hist_df.to_csv(history_file)

# model.save(model_file)