#!/usr/bin/env python3

import os.path
import numpy as np
import scipy.misc

image_file = './img_samples/test.jpg'
currentFrame = 1
save_path = './processed/test_' + str(currentFrame) + '.jpg'
image_shape = (300, 300)

#image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
image = scipy.misc.imread(image_file)

scipy.misc.imsave(save_path, image)
