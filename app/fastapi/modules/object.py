import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import modules.globals as g
import os

import modules.log as log

class Detector:
    def __init__(self):
        self.name = "object"
        log.logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        return

    def get_name(self):
        return self.name

    # runs yolov3 object detection
    def detect(self, fi, fo, args):
        
        log.logger.debug("Reading {}".format(fi))
        image = cv2.imread(fi)
        bbox, label, conf = cv.detect_common_objects(image)

        if not args['delete']:
            out = draw_bbox(image, bbox, label, conf)
            log.logger.debug("Writing {}".format(fo))
            cv2.imwrite(fo, out)

        detections = []

        for l, c, b in zip(label, conf, bbox):
            log.logger.debug ('-----------------------------------------')
            c = "{:.2f}%".format(c * 100)
            obj = {
                'type': l,
                'confidence': c,
                'box': b
            }
            log.logger.debug("{}".format(obj))
            detections.append(obj)

        if args['delete']:
            log.logger.debug("Deleting file {}".format(fi))
            os.remove(fi)

        details = {
            'input': l,
            'output': c,
            'box': b
        }

        return detections
