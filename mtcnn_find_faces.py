from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description="locate faces in the picture")

parser.add_argument('--photo', action='store_true', default=False, dest='isPhoto', help="to detect photo")
parser.add_argument('--webcam', action='store_true', default=False, dest='isWebcam', help="to detect video")
parser.add_argument('-f', '--file', dest='filename', help="file to detect faces")

args = parser.parse_args()


def draw_face_box(img_data, faces):
    
    pyplot.imshow(img_data)
    ax = pyplot.gca()
    
    for face in faces:
        
        x, y, w, h = face['box']
        box = Rectangle((x, y), w, h, fill=False, color='yellow')
        ax.add_patch(box)
        
        for key, value in face['keypoints'].items():
            
            point = Circle(value, radius=2, color='yellow')
            ax.add_patch(point)
            
    pyplot.show()
    
    
def extract_face(img_data, faces):
    
    face_list =[]
    
    for face in faces:
        x1, y1, w, h = face['box']
        x2, y2 = x1+w, y1+h
        
        face_list.append(img_data[y1:y2, x1:x2])
    
    return face_list

def check_argument(args):
    
    if args.isPhoto == args.isWebcam:
        print("Plase slect --photo or --webcam, and not both")
        exit(123)
    else:
        return True
    

if check_argument(args):
    
    
    
    if args.isPhoto:
        
        detector = MTCNN()
        
        pixels = pyplot.imread(args.filename)

        faces = detector.detect_faces(pixels)

        draw_face_box(pixels, faces)
        
    else:

        capture = cv2.VideoCapture(0)
        
        detector = MTCNN()
        
        while(True):
             
            ret, frame = capture.read()
             
            faces = detector.detect_faces(frame)
            
            for face in faces:
                x, y, w, h = face['box']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            cv2.imshow('video', frame)
             
            if cv2.waitKey(1) == 27:
                break
         
        capture.release()
        cv2.destroyAllWindows()
                       

        
    
#filename = 'D:\\thantham\\NS_RadovanovicIvan.jpg'
# load image from file
#pixels = pyplot.imread(filename)
# create the detector, using default weights
#detector = MTCNN()
# detect faces in the image
#faces = detector.detect_faces(pixels)

#draw_face_box(pixels, faces)




