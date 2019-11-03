import datetime
from enum import Enum
from typing import Dict, List
import os 

import cv2
from PIL import Image
from cvlib.object_detection import draw_bbox

import logging
logger = logging.getLogger(__name__)

class Detectors:
    def __init__(self):
        self.detectors_dict = dict()
                
    def add(self, model: object):
        self.detectors_dict[model.get_name()] = model

    def init(self, model_name: str):
        if model_name in self.detectors_dict:
            self.detectors_dict[model_name].init()
        else:
            error = 'detector with name {} not found'.format(model_name)
            logger.info(error)
            raise Exception(error)

    def init_all(self):
        for model in self.detectors_dict:
            self.detectors_dict[model].init()

    def detect(
            self, 
            name: str, 
            image_cv
        ):
        if name in self.detectors_dict:
            return self.detectors_dict[name].detect(image_cv)
        else:
            error = 'detector with name {} not found'.format(name)
            logger.info(error)
            raise Exception(error)

    def get(self):
        return self.detectors_dict.keys()

    def run(
            self,
            fip: str,
            ext: str,
            models_list: List[str],
            input_delete: bool,
            bbox_save: bool
        ):
    
        executed_succesfully = []
        failed = []

        start_total = datetime.datetime.now()
        args = dict()
        
        args['gender'] ="male"

        fi = fip + ext

        logger.debug("Reading {}".format(fi))
        #image_pil = Image.open(fi)
        image_cv = cv2.imread(fi)
        #image_pil = Image.fromarray(image_cv) # convert opencv frame (with type()==numpy) into PIL Image
                
        response_list = []
        
        for model_item in models_list: 
            try:
                start = datetime.datetime.now()
                fo = fip + '-' + model_item + ext
                bbox, label, conf = self.detect(model_item, image_cv)
                stop = datetime.datetime.now()
                elapsed_time = stop - start
                logger.info('detection took {}'.format(elapsed_time))

                details = {
                    "fi": fi,
                    "fo": fo,
                    "detection_time": elapsed_time
                }

                response_item = {
                    "model": model_item,
                    "details": details,
                    "error": None,
                    "model_response": {
                        "bbox": bbox,
                        "label": label,
                        "conf": conf    
                    }
                }
                response_list.append(response_item)

                executed_succesfully.append(model_item)

                if bbox_save == True:
                    tmp = image_cv.copy()
                    out = draw_bbox(tmp, bbox, label, conf, write_conf=True)
                    logger.debug("saving bbox image to {}".format(fo))
                    cv2.imwrite(fo, out)
    
            except Exception as error:
                logger.error('exception: {}'.format(error))
                response_item = {
                    "model": model_item,
                    "details": None,
                    "error": "invalid model: {}".format(model_item),
                    "model_response": None
                }
                response_list.append(response_item)
                failed.append(model_item)
                
        stop_total = datetime.datetime.now()
        elapsed_time_total = stop_total - start_total
        logger.info('TOTAL detection took {}'.format(elapsed_time_total))

        if input_delete == True:
            logger.debug("Deleting input file {}".format(fi))
            os.remove(fi)

        return executed_succesfully, failed, elapsed_time_total, response_list
    
