from torchvision import models, transforms
import torch
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt 
import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import glob, os, random, re
import pandas as pd 
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as Layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

import cv2

cardboard = glob.glob("C:\\Users\\Toktam\\Garbage Classification\\cardboard\\*.*")
glass = glob.glob("C:\\Users\\Toktam\\Garbage Classification\\glass\\*.*")
metal = glob.glob("C:\\Users\\Toktam\\Garbage Classification\\metal\\*.*")
paper = glob.glob("C:\\Users\\Toktam\\Garbage Classification\\paper\\*.*")
plastic = glob.glob("C:\\Users\\Toktam\\Garbage Classification\\plastic\\*.*")
trash = glob.glob("C:\\Users\\Toktam\\Garbage Classification\\trash\\*.*")

# definiting dataset and labels
dataset = []
labels = []


# load the images from cardboard, add them to dataset and label them as 0 
for img in tqdm(cardboard):   
    image=tf.keras.preprocessing.image.load_img(img)
    image=np.array(image)
    dataset.append(image)
    labels.append(0)

for img in tqdm(glass):   
    image=tf.keras.preprocessing.image.load_img(img)
    image=np.array(image)
    dataset.append(image)
    labels.append(1)
    
for img in tqdm(metal):   
    image=tf.keras.preprocessing.image.load_img(img)
    image=np.array(image)
    dataset.append(image)
    labels.append(2)
    
for img in tqdm(paper):   
    image=tf.keras.preprocessing.image.load_img(img)
    image=np.array(image)
    dataset.append(image)
    labels.append(3)
    
for img in tqdm(plastic):   
    image=tf.keras.preprocessing.image.load_img(img)
    image=np.array(image)
    dataset.append(image)
    labels.append(4)
    
for img in tqdm(trash):   
    image=tf.keras.preprocessing.image.load_img(img)
    image=np.array(image)
    dataset.append(image)
    labels.append(5)

    
dataset = np.array(dataset)
labels = np.array(labels)

# data augmentation by training image generator
dataAugmentaion = ImageDataGenerator(rescale=1./255,
                                      zoom_range=0.2,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      validation_split=0.2
                                      )

train_data = dataAugmentaion.flow_from_directory("C:\\Users\\Toktam\\Garbage Classification",
                                                     target_size=(384, 512),
                                                     batch_size=32,
                                                     class_mode = 'categorical',
                                                     subset = 'training')

                                    
val_data = dataAugmentaion.flow_from_directory("C:\\Users\\Toktam\\Garbage Classification",
                                                     target_size=(384, 512),
                                                     batch_size=32,
                                                     class_mode = 'categorical',
                                                     subset = 'validation')

labels = (train_data.class_indices)
labels = dict((v,k) for k,v in labels.items())

print(labels)


tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = 'logs',
      histogram_freq = 1,
      profile_batch = '1,100'
)

model = Sequential([
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(384, 512, 3)),
    MaxPooling2D(pool_size=2),
    Layers.Dropout(0.5),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Layers.Dropout(0.5),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Layers.Dropout(0.5),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# training the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(train_data, epochs=15, 
                    validation_data=val_data)
					
def predict(image_path):
    resnet = models.resnet101(pretrained=True)

    img = Image.open(image_path)

    pred = model.predict(img)
	
    return pred[0]