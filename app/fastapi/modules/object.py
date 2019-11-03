import cvlib as cv
import cv2
import numpy as np
import modules.globals as g
import modules.utils as utils
import os

import logging
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.name = "object"
        logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        return

    def get_name(self):
        return self.name

    # runs yolov3 object detection
    def detect(
            self, 
            image_cv
        ):
        
        bbox, label, conf = cv.detect_common_objects(image_cv)

        for l, c, b in zip(label, conf, bbox):
            logger.debug("type={}, confidence={:.2f}%, box={}".format(l,c,b))

        return bbox, label, conf
