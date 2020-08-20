import os
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import sys

from progress.bar import Bar
import argparse

parser = argparse.ArgumentParser(description="Produce image data, age and gender as .npy to training data")

parser.add_argument('-d', '--dir', required=True, default='utkfaces', dest="dir",help="Directory containing images with namely lebelled")
parser.add_argument('-s', '--imgsize', default=224, type=int, dest="size",help="target image size to load")
parser.add_argument('-g', '--gender', required=True, default='gender.npy', dest="gender",help="gender label to save as .npy")
parser.add_argument('-a', '--age', required=True, default='gender.npy', dest="age",help="age label to save .npy")
parser.add_argument('-i', '--imgdata', required=True, default='imgdata.npy', dest="imgdata",help="img data to save .npy")
parser.add_argument('-m', '--mode', dest='colormode', default='rgb',help="color mode of photo rgb or grayscale")

args = parser.parse_args()

#images_dir = "D:\\thantham\\imgpro\\utkfaces"
#img_data_npy = "training_image_data.npy"
#age_npy = "training_age.npy"
#gender_npy = "training_gender.npy"
#size = 224

images_dir = args.dir
img_data_npy = args.imgdata
age_npy = args.age
gender_npy = args.gender
size = args.size

def extract_age_gender(text):
    
    age = -1
    gender = -1
    
    if text[1] == "_":
        age = int(text[0])
        gender = int(text[2])
    elif text[2] == "_":
        age = int(text[0:2])
        gender = int(text[3])
    else:
        age = int(text[0:3])
        gender = int(text[4])
        
    return age, gender


def get_filewithpath(path):
    
    files = os.listdir(images_dir)
    
    file_list = []
    
    for file in files:
        file_list.append(images_dir+"\\"+file) 
    
    return file_list

def get_label_from_imagelist(image_dir):
    
    file_list = os.listdir(image_dir)
    
    print("number of images: ", len(file_list))
    
    Bar()
    age = []
    gender = []
    
    bar = Bar("getting label from image names", max=len(file_list))
    for file in file_list:
        
       age_i, gender_i = extract_age_gender(file)
       age.append(age_i)
       gender.append(gender_i)
       
       bar.next()
    
    bar.finish()
    return age, gender

def get_image_data_from_dir(imagedir_path):
    
    imagefile_list = get_filewithpath(imagedir_path)
    
    if args.colormode == 'rgb' :
        channel = 3
    else:
        channel = 1
        
    image_data = np.empty(shape = (len(imagefile_list),size, size, channel))
    
    print("images are being resized to: ", image_data.shape)
    
    bar = Bar("getting image data from face images", max=len(imagefile_list))
    for i in range(len(imagefile_list)):
        image_data[i] = img_to_array(load_img(imagefile_list[i], color_mode=args.colormode, target_size=(size,size)))
        
        bar.next()
    
    bar.finish()
    return image_data
        
age, gender = get_label_from_imagelist(images_dir)
image_data = get_image_data_from_dir(images_dir)

np.save(age_npy, age)
print("save age data as numpy array to: ", age_npy, "data size: ", sys.getsizeof(age)," Bytes")
np.save(gender_npy, gender)
print("save gender data as numpy array to: ", gender_npy, "data size: ", sys.getsizeof(gender)," Bytes")
np.save(img_data_npy, image_data)
print("save image data numpy array to: ", img_data_npy, "data size: ", sys.getsizeof(image_data)," Bytes")

