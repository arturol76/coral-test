from modules.detectors import DetectorResponse

import logging
logger = logging.getLogger(__name__)

class DetectorBase:
    def __init__(self, name: str):
        self.name = name
        logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        raise NotImplementedError()

    def get_model_name(self):
        return self.name

    # runs yolov3 object detection
    def detect(
            self, 
            image_cv
        ) -> DetectorResponse:
        
        raise NotImplementedError()