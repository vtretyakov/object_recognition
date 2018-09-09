#!/usr/bin/env python2
import os.path
import numpy as np
import scipy.misc
import tensorflow as tf
import cv2
#from keras.models import load_model


class Classifier(object):
    def __init__(self):
        MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

        # Path to frozen detection graph.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        self.model = None
        self.width = 0
        self.height = 0
        self.channels = 3

        # Load a frozen model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)
            # Input and output Tensors for detection_graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        image_np = np.asarray(image, dtype="uint8")
        image_np_expanded = np.expand_dims(image_np, axis=0)

        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        img_cnt = 1
        tl_image = None
        for idx, classID in enumerate(classes):
            if classID == 10:
                print ("found traffic light", scores[idx])
                if scores[idx] > 0.30:

                    nbox = boxes[idx]

                    height = image.shape[0]
                    width = image.shape[1]

                    box = np.array([nbox[0]*height, nbox[1]*width, nbox[2]*height, nbox[3]*width]).astype(int)
                    tl_image = image[box[0]:box[2], box[1]:box[3]]
                    img_out = cv2.cvtColor(tl_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite('img_samples/simulator/classified' + str(img_cnt) + '.jpg', img_out)
                    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
                    img_cnt += 1
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('img_samples/simulator/classified_box.jpg', image)
        return tl_image

# Execute `main()` function
if __name__ == '__main__':
    image_file = './img_samples/simulator/6.png'
    image = scipy.misc.imread(image_file)
    classifier = Classifier()
    classifier.get_classification(image)
