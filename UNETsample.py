"""
U-Net for Biomedical Image Segmentation
Overview
U-Net is a convolutional neural network designed for image segmentation, particularly in biomedical contexts. It has a U-shaped architecture with a contracting path to capture context and an expansive path for precise localization.

Steps to Implement U-Net
1. Install Required Libraries
First, install TensorFlow and TensorFlow Datasets using the following command:

$ pip install tensorflow tensorflow_datasets

2. Load and Preprocess the Dataset
Load a biomedical dataset, such as the Oxford-IIIT Pet Dataset, and preprocess the images and segmentation masks to a consistent size.

3. Define the U-Net Model
Create a U-Net model with an encoder-decoder architecture. The encoder captures the context of the image, while the decoder reconstructs the image to its original size with precise localization.

4. Compile and Train the Model
Compile the model with an appropriate optimizer and loss function, then train it on the dataset for a number of epochs.

5. Visualize the Results
After training, visualize the model’s predictions to evaluate its performance on the segmentation task.

Conclusion
Using U-Net for biomedical image segmentation involves installing necessary libraries, loading and preprocessing the dataset, defining the U-Net architecture, compiling and training the model, and visualizing the results. This approach can be adapted to various biomedical datasets to achieve high-accuracy segmentation.
"""

#!pip install tensorflow tensorflow_datasets

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the Oxford-IIIT Pet Dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
train_dataset = dataset['train'].map(lambda x: (tf.image.resize(x['image'], (128, 128)), tf.image.resize(x['segmentation_mask'], (128, 128))))
train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Define the U-Net model
def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')(c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile the model
model = unet_model(output_channels=3)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=20)

# Visualize some predictions
def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

for image, mask in train_dataset.take(1):
    pred_mask = model.predict(image)
    display([image[0], mask[0], pred_mask[0]])