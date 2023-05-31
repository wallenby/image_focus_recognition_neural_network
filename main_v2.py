
# Libraries 
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense

devices = tf.config.list_physical_devices()

# Getting Paths
focus_folder = 'focused_path'
unfocus_folder = 'not_focused_path'


# Arrays for 
focus_images = []
unfocus_images = []


for filename in os.listdir(focus_folder):
    img = cv2.imread(os.path.join(focus_folder, filename))
    if img is not None:
        focus_images.append(img)

for filename in os.listdir(unfocus_folder):
    img = cv2.imread(os.path.join(unfocus_folder, filename))
    if img is not None:
        unfocus_images.append(img)


img_size = 256

focus_images_resized = []
unfocus_images_resized = []

for img in focus_images:
    img_resized = cv2.resize(img, (img_size, img_size))
    focus_images_resized.append(img_resized)

for img in unfocus_images:
    img_resized = cv2.resize(img, (img_size, img_size))
    unfocus_images_resized.append(img_resized)



focus_labels = np.ones(len(focus_images_resized))
unfocus_labels = np.zeros(len(unfocus_images_resized))
images = np.concatenate([focus_images_resized, unfocus_images_resized], axis=0)
labels = np.concatenate([focus_labels, unfocus_labels], axis=0)

p = np.random.permutation(len(images))
images = images[p]
labels = labels[p]

images = images.astype('float32') / 255.0

train_ratio = 0.8
train_size = int(len(images) * train_ratio)

train_images = images[:train_size]
train_labels = labels[:train_size]

test_images = images[train_size:]
test_labels = labels[train_size:]




# Create a TensorFlow session with GPU support
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size,img_size,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels))

model.summary()

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Loss:', test_loss, '   Test accuracy:', test_acc)



# Saving model (v2)
model.save("model_v2.h5")



# Looping to keep receiving and analyzing new images
while(True):
    
    path = input("Enter your path: ")
    new_image = cv2.imread(path)
    new_image_resized = cv2.resize(new_image, (img_size, img_size))
    new_image_resized = new_image_resized.astype('float32') / 255.0
    new_image_resized = np.expand_dims(new_image_resized, axis=0)

    prediction = model.predict(new_image_resized)
    print("Predict. :", prediction)
    
    if prediction < 0.5:
        print('Out-of-focus')
    else:
        print('In-focus')
