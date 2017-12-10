def create_mask(img):
    """
	A function to ...
	"""
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def segment(img):
    """
	A function to...
	"""
    mask = create_mask(img)
    output = cv2.bitwise_and(img, img, mask=mask)
    return output
