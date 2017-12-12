# import required packages
import os
import cv2
from keras.preprocessing import image
from img_preprocessing import preprocess_image

# set the paths for root project folder, base training data folder, and test data folder
PATH = "/home/mike/kaggle"
TRAINING_FOLDER = os.path.join(PATH, "train")
TEST_IMAGES = os.path.join(PATH, "test")

# create a list of all images in the test set
pic_list = image.list_pictures(TEST_IMAGES)
if not os.path.exists(os.path.join(PATH, "test_mod")):
    # create a directory for preprocessed test images
    os.mkdir(path=os.path.join(PATH, "test_mod"))
    print("Created the 'test_mod' directory")
else:
    # if directory already exists...
    print("The 'test_mod' directory already exists")

# target path to write to
TARGET_PATH = os.path.join(PATH, "test_mod")
for pic in pic_list:
    # read in the image
    img = cv2.imread(pic)
    # create image mask and segment the original image using the mask
    img = preprocess_image(img)
    # write the segmented image to the TARGET_PATH
    cv2.imwrite(os.path.join(TARGET_PATH, os.path.basename(pic)), img)
