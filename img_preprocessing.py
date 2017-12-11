def preprocess_image(img):
    """
    A function to convert the colour image into the segmented image for better feature extraction.
    """

    # Convert the Image from BGR to HSV
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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
