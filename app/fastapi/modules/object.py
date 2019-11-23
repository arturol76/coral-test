import cvlib as cv
import cv2
import numpy as np
import modules.utils as utils
import os

import modules.detectors as detectors_model

import logging
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.name = "object"
        logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        return

    def get_model_name(self):
        return self.name

    # runs yolov3 object detection
    def detect(
            self, 
            image_cv
        ) -> detectors_model.DetectorResponse:
        
        bbox, label, conf = cv.detect_common_objects(image_cv)

        model_response = detectors_model.DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c)

        return model_response
