#!/usr/bin/env python2
import numpy as np
import scipy.misc
import cv2


def crop_n_blur(image):
    row = 2
    col = 6
    img = image.copy()
    img = img[row:-row, col:-col, :]
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def standardize_input(image):
    standard_im = np.copy(image)
    standard_im = cv2.resize(standard_im, (32, 32))
    standard_im = crop_n_blur(standard_im)
    return standard_im

def slice_image(image):
    img = image.copy()
    shape = img.shape
    slice_height = shape[0]/3
    upper = img[0:slice_height, :, :]
    middle = img[slice_height:2*slice_height, :, :]
    lower = img[2*slice_height:3*slice_height, :, :]
    return upper, middle, lower

def findNonZero(image):
    rows, cols, _ = image.shape
    counter = 0

    for row in range(rows):
        for col in range(cols):
            pixel = image[row, col]
            if sum(pixel) != 0:
                counter = counter + 1
    return counter


def get_color_hsv(image):
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

def get_color_lab(image_bgr):
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l = image_lab.copy()
    # set a and b channels to 0
    l[:, :, 1] = 0
    l[:, :, 2] = 0

    # cv2.imshow('LAB image l', l)
    # cv2.waitKey(0)

    std_l = standardize_input(l)
    # cv2.imshow('std_l', std_l)
    # cv2.waitKey(0)

    red_slice, yellow_slice, green_slice = slice_image(std_l)
    # cv2.imshow('red_slice', red_slice)
    # cv2.waitKey(0)
    # cv2.imshow('yellow_slice', yellow_slice)
    # cv2.waitKey(0)
    # cv2.imshow('green_slice', green_slice)
    # cv2.waitKey(0)

    y, x, c = red_slice.shape
    px_sums = []
    color = ['RED', 'YELLOW', 'GREEN']
    px_sums.append(np.sum(red_slice[0:y, 0:x, 0]))
    px_sums.append(np.sum(yellow_slice[0:y, 0:x, 0]))
    px_sums.append(np.sum(green_slice[0:y, 0:x, 0]))

    max_value = max(px_sums)
    max_index = px_sums.index(max_value)

    return color[max_index]

# Execute `main()` function
if __name__ == '__main__':
    image_file = './img_samples/simulator/classified6.jpg'
    image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)

    #image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    #image = scipy.misc.imread(image_file)

    tl_color = get_color_lab(image_bgr)
    print ('The light is ' + tl_color)

    cv2.imshow('Sample', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()