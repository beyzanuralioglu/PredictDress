# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:27:35 2021

@author: PC
"""
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers

trainfile=r'C:\Users\PC\Desktop\dataset\train'
validationfile=r'C:\Users\PC\Desktop\dataset\validation'


image_size = (180, 180)
batch_size = 10

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    trainfile,
    validation_split=0.01,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validationfile,
    validation_split=0.9,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

model = tf.keras.Sequential(
    [
     layers.Conv2D(32, 3, activation='relu'),
     layers.Conv2D(32, 3, activation='relu'),
     layers.Conv2D(32, 3, activation='relu'),
     layers.MaxPooling2D(),
     layers.Conv2D(32, 3, activation='relu'),
     layers.Conv2D(32, 3, activation='relu'),
     layers.Conv2D(32, 3, activation='relu'),
     layers.MaxPooling2D(),
     layers.Flatten(),
     layers.Dense(128, activation='relu'),
     layers.Dense(2)    
     ])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=6
)

model.save("result_1")















