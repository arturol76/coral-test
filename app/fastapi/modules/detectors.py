import datetime
import logging
import os
from enum import Enum
from typing import Dict, List

import cv2
import modules.utils as utils
from cvlib.object_detection import draw_bbox
from PIL import Image
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DetectorRecord:
    def __init__(
            self,
            bbox:       list,
            label:      str,
            conf:       float,
            model_name: str
        ):
        self.bbox=bbox
        self.label=label
        self.conf=conf
        self.model_name=model_name

    def print(self):
        logger.debug("label={}, conf={:.4f}, model={}, box={}".format(self.label,self.conf,self.model_name,self.bbox))

@dataclass
class DetectorResponse:
    def __init__(
            self,
            model_name: str
        ):
        self.model_name = model_name
        self.data: List[DetectorRecord] = []
        logger.debug('{}: new instance'.format(type(self).__name__))
            
    def add(
            self,
            bbox:       list,
            label:      str,
            conf:       float
        ):
        logger.debug('{}: adding record'.format(type(self).__name__))
        record = DetectorRecord(bbox,label,conf,self.model_name)
        record.print()
        self.data.append(record)

    def print(self):
        for record in self.data:
            record.print()

    def get_blcm_vectors(self):
        b = []
        l = []
        c = []
        m = []
        for item in self.data:
            b.append(item.bbox)
            l.append(item.label)
            c.append(item.conf)
            m.append(item.model_name)

        return b,l,c,m

@dataclass
class RunModelResponse:
    def __init__(
            self,
            file_in: str,
            file_out: str,
            detection_time: str,
            error: str,
            model_response: DetectorResponse
        ):
        self.file_in = file_in
        self.file_out = file_out
        self.detection_time=detection_time
        self.error=error
        self.model_response=model_response
        logger.debug('{}: new instance'.format(type(self).__name__))

class RunBatchResponse(BaseModel):
    file_in:        str = ""
    executed_ok:    List[str] = []
    failed:         List[str] = []
    total_time:     str = ""
    response_list:  List[RunModelResponse] = []

class Detectors:
    def __init__(self):
        self.detectors_dict = dict()
                
    def add(self, model: object):
        self.detectors_dict[model.get_model_name()] = model

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
        ) -> DetectorResponse:

        if name in self.detectors_dict:
            return self.detectors_dict[name].detect(image_cv)
        else:
            error = 'detector with name {} not found'.format(name)
            logger.error(error)
            raise Exception(error)

    def get(self):
        return self.detectors_dict.keys()

    def run(
            self,
            fip: str,               #input file with path
            models_list: List[str],
            image_save: bool,
            bbox_save: bool
        ) -> RunBatchResponse:
    
        batch_response = RunBatchResponse()
        
        start_total = datetime.datetime.now()

        logger.debug("reading {}".format(fip))
        #image_pil = Image.open(fi)
        image_cv = cv2.imread(fip)
        #image_pil = Image.fromarray(image_cv) # convert opencv frame (with type()==numpy) into PIL Image
                
        for model_to_be_executed in models_list: 
            try:
                start = datetime.datetime.now()

                fop_no_ext, ext = os.path.splitext(fip)
                fop = fop_no_ext + '-' + model_to_be_executed + ext
                detector_response = self.detect(model_to_be_executed, image_cv)
                stop = datetime.datetime.now()
                elapsed_time = stop - start
                logger.info('detection took {}'.format(elapsed_time))

                run_detector_response = RunModelResponse(
                    fip,
                    fop,
                    elapsed_time,
                    "",
                    detector_response
                )

                batch_response.executed_ok.append(model_to_be_executed)

                if bbox_save == True:
                    tmp = image_cv.copy()
                    #out = draw_bbox(tmp, bbox, label, conf, write_conf=True)
                    
                    b,l,c,m = run_detector_response.model_response.get_blcm_vectors()
                    out = utils.draw_bbox2(tmp, b, l, c, write_conf=True)
                    logger.debug("saving bbox image to {}".format(fop))
                    cv2.imwrite(fop, out)
    
            except Exception as error:
                logger.error('exception: {}'.format(error))
                detector_response.error = "invalid model: {}, error: {}".format(model_to_be_executed, error)
                batch_response.failed.append(model_to_be_executed)

            finally:
                batch_response.response_list.append(run_detector_response)
        
        if image_save == False:
            logger.debug("Deleting input file {}".format(fip))
            os.remove(fip)

        stop_total = datetime.datetime.now()
        batch_response.total_time = stop_total - start_total
        logger.info('TOTAL detection took {}'.format(batch_response.total_time))

        return batch_response

    def merge(
            self,
            batch_response: RunBatchResponse
        ) -> DetectorResponse:
    
        logger.debug('merging...')

        start_total = datetime.datetime.now()
        
        merged = DetectorResponse("merged")

        for response_item in batch_response.response_list:
            merged.data += response_item.model_response.data

        for record in merged.data:
            print("merged: {},{},{},{}".format(record.label,record.conf,record.model_name,record.bbox))

        stop_total = datetime.datetime.now()
        total_time = stop_total - start_total
        logger.info('TOTAL detection took {}'.format(total_time))

        return merged

    def nms(
            self,
            merged:         DetectorResponse,
            conf_threshold: float,
            nms_threshold:  float
        ) -> DetectorResponse:

        start_total = datetime.datetime.now()

        b,l,c,m = merged.get_blcm_vectors()
        indices = cv2.dnn.NMSBoxes(b, c, conf_threshold, nms_threshold)

        nms_out = DetectorResponse("nms")

        for i in indices:
            i = i[0]
            nms_out.add(b[i], l[i], c[i])

        for record in nms_out.data:
            print("nms: {},{},{},{}".format(record.label,record.conf,record.model_name,record.bbox))

        stop_total = datetime.datetime.now()
        total_time = stop_total - start_total
        logger.info('TOTAL detection took {}'.format(total_time))

        return nms_out
