#!/usr/bin/env python3
import os.path
import numpy as np
import scipy.misc
import tensorflow as tf
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

    def init_classifier(self, model, width, height, channels=3):
        self.width = width
        self.height = height
        self.model = model
        self.channels = channels
        # necessary work around to avoid troubles with keras
        #self.graph = tf.get_default_graph()

    def get_classification(self, image):
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
        print(boxes, scores, classes, num)

# Execute `main()` function
if __name__ == '__main__':
    image_file = './img_samples/test.jpg'
    image = scipy.misc.imread(image_file)
    classifier = Classifier()
    classifier.get_classification(image)