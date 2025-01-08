# src/data_preprocessing.py
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

def preprocess_data(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    print(f"Loading training data from: {train_dir}")
    print(f"Loading validation data from: {val_dir}")

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                       height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size,
                                                        batch_size=batch_size, class_mode='binary')

    val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size,
                                                    batch_size=batch_size, class_mode='binary')
    return train_generator, val_generator