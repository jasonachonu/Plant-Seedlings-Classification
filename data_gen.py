import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical

# root project path
PATH = "/home/mike/kaggle"
# path to training data
TRAINING_FOLDER = os.path.join(PATH, "train")
# path to preprocessed training data
TRAIN_MOD = os.path.join(PATH, "train_mod")
# path to test data
TEST_IMAGES = os.path.join(PATH, "test")
# path to preprocessed test data
TEST_MOD = os.path.join(PATH, "test_mod")
# input image size for rescaling and passing to the network
INPUT_SIZE = 50

# target labels for the 12 plant seedling species
labels = os.listdir(TRAINING_FOLDER)

# create DataFrame with the path to each training image, the encoded label, and the text label
original = []
for label_id, label in enumerate(labels):
    for file in os.listdir(os.path.join(TRAINING_FOLDER, label)):
        original.append(['train/{}/{}'.format(label, file), label_id, label])
original = pd.DataFrame(original, columns=['file', 'label_id', 'label'])

# create DataFrame with the path to each preprocessed training image
mod = []
for label_id, label in enumerate(labels):
    for file in os.listdir(os.path.join(TRAINING_FOLDER, label)):
        mod.append('train_mod/{}/{}'.format(label, file))
mod = pd.DataFrame(mod, columns=['mod_file'])

# horizontally concatenate the 2 DataFrames
frames = [original, mod]
train = pd.concat(frames, axis=1)


def read_img(file_path, size):
    """
    A function to read in an input image, rescale the image to a set target size, and convert
    the image to an array.
    :param file_path: The path to the image in either the training or test set folder
    :param size: the target image size to rescale to
    :return: Returns the rescaled image as an array
    """
    img = image.load_img(os.path.join(PATH, file_path), target_size=size)
    img = image.img_to_array(img)
    return img


def create_training_set(size, use_modified=True):
    """
    A function to create the set of training examples
    :param use_modified: Boolean indicating whether we want to use the preprocessed images or the originals
    :param size: the target image size to rescale to
    :return: returns the feature matrix, x, and the target labels, y
    """
    x = np.zeros((len(train), size, size, 3), dtype='float32')
    y = to_categorical(train['label_id'])
    if use_modified:
        for i, fn in enumerate(train['mod_file']):
            img = read_img(fn, (size, size))
            img = img / 255
            img = np.expand_dims(img, axis=0)
            x[i] = img
    else:
        for i, fn in enumerate(train['file']):
            img = read_img(fn, (size, size))
            img = img / 255
            img = np.expand_dims(img, axis=0)
            x[i] = img
    return x, y


# assemble the training data
X, targets = create_training_set(size=INPUT_SIZE, use_modified=False)
np.save(file='X', arr=X)
np.save(file='targets', arr=targets)
