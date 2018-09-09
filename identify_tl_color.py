#!/usr/bin/env python2
import numpy as np
import scipy.misc
import cv2


def crop_n_blur(image):
    row = 2
    col = 7
    img = image.copy()
    img = img[row:-row, col:-col, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    standard_im = crop_n_blur(standard_im)
    return standard_im

def findNonZero(image):
    rows, cols, _ = image.shape
    counter = 0

    for row in range(rows):
        for col in range(cols):
            pixel = image[row, col]
            if sum(pixel) != 0:
                counter = counter + 1
    return counter


def get_color(image):
    color = 'none'

    image = standardize_input(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:, :, 1])  # Sum the brightness values
    shape = image.shape
    area = shape[0] * shape[1]
    avg_saturation = sum_saturation / area  # Find the average
    sat_low = int(avg_saturation * 1.3)
    val_low = 140

    print(avg_saturation, sat_low)

    # Green
    lower_green = np.array([70, sat_low, val_low])
    upper_green = np.array([100, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_result = cv2.bitwise_and(image, image, mask=green_mask)

    # Yellow
    lower_yellow = np.array([10, sat_low, val_low])
    upper_yellow = np.array([60, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(image, image, mask=yellow_mask)

    # Red
    lower_red = np.array([150, sat_low, val_low])
    upper_red = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result = cv2.bitwise_and(image, image, mask=red_mask)

    sum_green = findNonZero(green_result)
    sum_yellow = findNonZero(yellow_result)
    sum_red = findNonZero(red_result)

    print ("sums g y r:", sum_green, sum_yellow, sum_red)

    if sum_red >= sum_yellow and sum_red >= sum_green:
        color = 'Red'
        print("Red")
    elif sum_yellow >= sum_green:
        color = 'Yellow'
        print("Yellow")
    else:
        color = 'Green'
        print("Green")

    cv2.imshow('Test image', image)
    cv2.waitKey(0)
    cv2.imshow('HSV image', hsv)
    cv2.waitKey(0)
    cv2.imshow('green_result', green_result)
    cv2.waitKey(0)
    return color


# Execute `main()` function
if __name__ == '__main__':
    image_file = './img_samples/simulator/classified7.jpg'
    image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)

    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l = image_lab.copy()
    # set a and b channels to 0
    l[:, :, 1] = 0
    l[:, :, 2] = 0

    cv2.imshow('LAB image l', l)
    cv2.waitKey(0)

    #image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    #image = scipy.misc.imread(image_file)
    #tl_color = get_color(image)
    #print (tl_color)
    cv2.destroyAllWindows()