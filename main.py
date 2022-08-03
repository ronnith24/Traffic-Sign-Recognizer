import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers.pooling import MaxPooling2D

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Loads image data from directory `data_dir`.
    Assumes `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory there should be some
    number of image files.
    Return tuple `(images, labels)`. `images` is a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` is
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []
    # looping over every category
    for category in range(NUM_CATEGORIES):
        current_path = os.path.join(data_dir, str(category))
        # getting all the images paths
        img_paths = [img_path for img_path in os.listdir(current_path)]
        for img_path in img_paths:
            path = os.path.join(current_path, img_path)
            img = cv2.imread(path)
            # resizing the image
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            labels.append(category)
    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. 
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
              kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # flattening for connection to the Dense layer and adding dropout to regularize the model
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))

    # chose a softmax activation function of classification
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # optimizeing with adam while keeping track of accuracy
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    main()
