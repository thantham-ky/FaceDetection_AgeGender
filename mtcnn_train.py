from keras.applications import MobileNetV2, Xception
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D, Input

from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from keras.optimizers import Adam

import numpy as np
import pandas as pd


image_data_file = "D:\\thantham\\imgpro\\imgdata.npy"
age_data_file = "D:\\thantham\\imgpro\\age.npy"
gender_data_file = "D:\\thantham\\imgpro\\gender.npy"

history_file = "D:\\thantham\\imgpro\\history.csv"
model_file = "D:\\thantham\\imgpro\\model.h5"

print("load data......", end="\n\n\n")
image_data = np.load(image_data_file)
gender_data = np.load(gender_data_file)
age_data = np.load(age_data_file)


print("training data shape: ", image_data.shape[1:])
print("number of training: ", image_data.shape[0])

print("train and test set splitting..... ", end="\n\n\n")
#y = np.array([gender_data, age_data]).transpose()
y = np.array([gender_data, age_data]).transpose()
x_train, x_test, y_train, y_test = train_test_split(image_data, y, test_size=0.2)

print("Model constructing.....", end="\n\n\n")

input_layer = Input(shape=x_train.shape[1:])

x = MobileNetV2(include_top=False)(input_layer)
#x = GlobalAveragePooling2D()(x)

gend_branch = Flatten()(x)
gend_branch = Dense(1024, activation='relu')(gend_branch)
gender_out = Dense(1, activation='sigmoid', name='gender_out')(gend_branch)

age_branch = Flatten()(x)
age_branch = Dense(1024, activation='relu')(age_branch)
age_out = Dense(1, activation='linear', name='age_out')(age_branch)

model = Model( inputs=input_layer, outputs=[gender_out, age_out])

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])


print("Fitting.....", end="\n\n\n")
history = model.fit(x_train, y=[y_train[:,0], y_train[:, 1]], 
                    validation_data= (x_test, [y_test[:,0], y_test[:, 1]]),
                    batch_size=32, epochs=5, workers=4)

hist_df = pd.DataFrame(history.history)
hist_df.to_csv(history_file)

model.save(model_file)
