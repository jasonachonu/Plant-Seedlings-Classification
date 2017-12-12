import numpy as np
import cv2


def preprocess_image(img):
    """
    A function to convert the colour image into the segmented image for better feature extraction.
    """
    
    #Convert the Image from RGB to BGR
    image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Convert the Image from BGR to HSV
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    # Creating a mask 
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    # Creating an Elliptical Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # Remove noise from the image
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Use mask to create a Segmented Image
    output = cv2.bitwise_and(img, img, mask=mask)
    return output


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
