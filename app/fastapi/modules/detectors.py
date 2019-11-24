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
class _DetectorTag:
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
        self.file_bbox = None
        self.data: List[_DetectorTag] = []
        logger.debug('{}: new instance'.format(type(self).__name__))
            
    def add(
            self,
            bbox:       list,
            label:      str,
            conf:       float,
            model_name: str
        ):
        logger.debug('{}: adding record'.format(type(self).__name__))
        record = _DetectorTag(bbox,label,conf,model_name)
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

    def draw_bbox_and_save(
            self,
            image_cv:       object,
            save_to_file:   bool,
            file_in:        str,
            color=None, 
            write_conf:     bool =True
        ) -> object:
        
        tmp = image_cv.copy()
        b,l,c,m = self.get_blcm_vectors()
        image_bbox = utils.draw_bbox2(tmp, b,l,c,color,write_conf)
        
        if save_to_file == True:
            fop_no_ext, ext = os.path.splitext(file_in)
            self.file_bbox = fop_no_ext + '-' + self.model_name + ext
            logger.debug("saving bbox image to {}".format(self.file_bbox))
            cv2.imwrite(self.file_bbox, image_bbox)
        else:
            self.file_bbox = None

        return  image_bbox

@dataclass
class RunDetectorResponse:
    def __init__(
            self,
            file_in: str,
            detection_time: str,
            error: str,
            model_response: DetectorResponse
        ):
        self.file_in = file_in
        self.detection_time=detection_time
        self.error=error
        self.model_response=model_response
        logger.debug('{}: new instance'.format(type(self).__name__))

@dataclass
class RunBatchResponse:
    def __init__(
            self,
            file_in:        str
        ):
        self.file_in = file_in
        self.executed_ok:    List[str] = []
        self.failed:         List[str] = []
        self.total_time:     str = ""
        self.response_list:  List[RunDetectorResponse] = []
        logger.debug('{}: new instance'.format(type(self).__name__))

    def add_ok(
            self,
            detection_time: str,
            detector_response: DetectorResponse
        ):
        logger.debug('{}: adding record'.format(type(self).__name__))
        record = RunDetectorResponse(self.file_in,detection_time,"",detector_response)
        self.response_list.append(record)
        self.executed_ok.append(detector_response.model_name)

    def add_failed(
            self,
            error:          str,
            detection_time: str,
            detector_response: DetectorResponse
        ):
        logger.debug('{}: adding record'.format(type(self).__name__))
        record = RunDetectorResponse(self.file_in,detection_time,error,detector_response)
        self.response_list.append(record)
        self.executed_ok.append(detector_response.model_name)


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
    
        batch_response = RunBatchResponse(fip)
        
        start_total = datetime.datetime.now()

        logger.debug("reading {}".format(fip))
        #image_pil = Image.open(fi)
        image_cv = cv2.imread(fip)
        #image_pil = Image.fromarray(image_cv) # convert opencv frame (with type()==numpy) into PIL Image
                
        for model_name in models_list: 
            try:
                start = datetime.datetime.now()
                detector_response = self.detect(model_name, image_cv)
                detector_response.draw_bbox_and_save(image_cv,bbox_save,fip,write_conf = True)
                stop = datetime.datetime.now()
                elapsed_time = stop - start
                logger.info('detection took {}'.format(elapsed_time))
    
            except Exception as error:
                logger.error('exception: {}'.format(error))
                error = "invalid model: {}, error: {}".format(model_name, error)
                batch_response.add_failed(error, elapsed_time, detector_response)

            finally:
                batch_response.add_ok(elapsed_time, detector_response)
        
        if image_save == False:
            logger.debug("Deleting input file {}".format(fip))
            os.remove(fip)

        stop_total = datetime.datetime.now()
        batch_response.total_time = stop_total - start_total
        logger.info('TOTAL detection took {}'.format(batch_response.total_time))

        #TEST
        merged, elapsed_time1 = self.merge(batch_response)
        nms, elapsed_time2 = self.nms(merged, 0.5, 0.8)
        nms.draw_bbox_and_save(image_cv,bbox_save,fip,write_conf = True)
        batch_response.add_ok(elapsed_time1+elapsed_time2, nms)

        return batch_response

    def merge(
            self,
            batch_response: RunBatchResponse
        ) -> (DetectorResponse, object):
    
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

        return merged, total_time

    def nms(
            self,
            merged:         DetectorResponse,
            conf_threshold: float,
            nms_threshold:  float
        ) -> (DetectorResponse, object):

        start_total = datetime.datetime.now()

        b,l,c,m = merged.get_blcm_vectors()
        indices = cv2.dnn.NMSBoxes(b, c, conf_threshold, nms_threshold)

        nms_out = DetectorResponse("nms")

        for i in indices:
            i = i[0]
            nms_out.add(b[i], l[i], c[i], m[i])

        for record in nms_out.data:
            print("nms: {},{},{},{}".format(record.label,record.conf,record.model_name,record.bbox))

        stop_total = datetime.datetime.now()
        total_time = stop_total - start_total
        logger.info('TOTAL detection took {}'.format(total_time))

        return nms_out, total_time
