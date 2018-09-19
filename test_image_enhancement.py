9#!/usr/bin/env python2

import cv2
import numpy as np

def clahe(bgr, gridsize=8):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def adjust_gamma(bgr, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(bgr, table)




# Execute `main()` function
if __name__ == '__main__':
    image_path = 'img_samples/test15.jpg'
    bgr = cv2.imread(image_path)
    #enhanced = clahe(bgr)
    enhanced = adjust_gamma(bgr, 0.5)
    cv2.imshow('enhanced',enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()