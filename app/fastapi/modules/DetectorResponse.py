import os
from typing import Dict, List

import cv2
import modules.utils as utils
from PIL import Image
from pydantic.dataclasses import dataclass

import logging
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
