# import required packages
import os
import cv2
from keras.preprocessing import image
from img_preprocessing import preprocess_image

# set the paths for root project folder, base training data folder, and test data folder
PATH = "/Users/michaelarango/Desktop/ml2-proj"
TRAINING_FOLDER = os.path.join(PATH, "train")
TEST_IMAGES = os.path.join(PATH, "test")

labels = os.listdir(TRAINING_FOLDER)

# create a list of all images in the test set
pic_list = image.list_pictures(TRAINING_FOLDER)

if not os.path.exists(os.path.join(PATH, "train_mod")):
    # create a directory for preprocessed test images
    os.mkdir(path=os.path.join(os.path.join(PATH, "train_mod")))
    print("Created the 'train_mod' directory")
else:
    # if directory already exists...
    print("The 'train_mod' directory already exists")

# target path to write to
TARGET_PATH = os.path.join(PATH, "train_mod")

for label in labels:
    os.makedirs(TARGET_PATH + "/{}".format(label), exist_ok=True)

for pic in pic_list:
    img = cv2.imread(pic)
    img = preprocess_image(img)
    SUB_PATH = pic.split('/')[-2] + '/' + pic.split('/')[-1]
    cv2.imwrite(os.path.join(TARGET_PATH, SUB_PATH), img)
