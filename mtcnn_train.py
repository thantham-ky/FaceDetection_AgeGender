from keras.applications import Xception, MobileNetV2
from keras.models import  Model
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.optimizers import RMSprop

from keras.utils import plot_model
from keras.utils.vis_utils import pydot
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


image_data_file = "F:\\Facedet_Kwang\\FaceDetection_AgeGender-master\\FaceDetection_AgeGender-master\\imgdata.npy"
age_data_file = "F:\\Facedet_Kwang\\FaceDetection_AgeGender-master\\FaceDetection_AgeGender-master\\age.npy"
gender_data_file = "F:\\Facedet_Kwang\\FaceDetection_AgeGender-master\\FaceDetection_AgeGender-master\\gender.npy"

history_file = "F:\\Facedet_Kwang\\FaceDetection_AgeGender-master\\FaceDetection_AgeGender-master\\history.csv"
model_file = "F:\\Facedet_Kwang\\FaceDetection_AgeGender-master\\FaceDetection_AgeGender-master\\model.h5"
model_checkpointfile = "F:\\Facedet_Kwang\\FaceDetection_AgeGender-master\\FaceDetection_AgeGender-master\\model_checkpoint.h5"

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
x_train, x_test, y_train, y_test = train_test_split(image_data, y, test_size=0.2)

datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, rescale=0.1)


print("Model constructing.....", end="\n\n\n")

callbacks = [ EarlyStopping(monitor='val_loss', patience=5, verbose=0), 
              ModelCheckpoint(model_checkpointfile, monitor='val_loss', save_best_only=True, verbose=0),
              ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]


base_model = Xception(input_shape=x_train.shape[1:], include_top=False, pooling='avg')
x = base_model.output

gend_branch = Dense(256, activation='relu')(x)
gend_branch = Dropout(0.5)(gend_branch)
gender_out = Dense(1, activation='sigmoid', name='gender_out')(gend_branch)


age_branch = Dense(256, activation='relu')(x)
age_branch = Dropout(0.5)(age_branch)
age_out = Dense(1, activation='linear', name='age_out')(age_branch)

model = Model(inputs=base_model.input, outputs=[gender_out, age_out])

model.compile(loss={'gender_out': 'binary_crossentropy', 'age_out': 'mse'}, optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

print("Fitting.....", end="\n\n\n")

# history = model.fit_generator(datagen.flow(x_train, y={'gender_out':y_train[:, 0],'age_out':y_train[:, 1]}, batch_size=32),
#                                             steps_per_epoch = len(x_train)/32,
#                                             validation_data= datagen.flow(x_test, y={'gender_out':y_test[:, 0],'age_out':y_test[:, 1]}, batch_size=32),
#                                             validation_steps=len(x_test)/32,
#                                             callbacks= callbacks,
#                                             epochs=100,
#                                             verbose=1,
#                                             workers=4)

# history = model.fit_generator(datagen.flow(x_train, [y_train[:,0], y_train[:, 1]]))

history = model.fit(x_train, {'gender_out':y_train[:, 0],'age_out':y_train[:, 1]}, 
                    validation_data= (x_test, {'gender_out':y_test[:, 0],'age_out':y_test[:, 1]}),
                    batch_size=32, epochs=100, workers=4, callbacks=callbacks)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(history_file)

model.save(model_file)