import cvlib as cv

from modules.detectors import DetectorResponse
from modules.DetectorBase import DetectorBase

import logging
logger = logging.getLogger(__name__)

class Detector(DetectorBase):
    def __init__(self):
        DetectorBase.__init__(self, "object")

    def init(self):
        return

    # runs yolov3 object detection
    def detect(
            self, 
            image_cv
        ) -> DetectorResponse:
        
        bbox, label, conf = cv.detect_common_objects(image_cv)

        model_response = DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c,self.get_model_name())

        return model_response
