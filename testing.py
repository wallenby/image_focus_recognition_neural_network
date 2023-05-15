import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np


model = keras.models.load_model('/Users/erikmenard/Desktop/ai_projects/focus_recognition_ai/focus_model.h5')

img_size = 256

while(True):
    
    path = input("Enter your path: ")
    new_image = cv2.imread(path)
    new_image_resized = cv2.resize(new_image, (img_size, img_size))
    new_image_resized = new_image_resized.astype('float32') / 255.0
    new_image_resized = np.expand_dims(new_image_resized, axis=0)

    prediction = model.predict(new_image_resized)
    if prediction < 0.5:
        print('The image is out of focus')
    else:
        print('The image is in focus')
