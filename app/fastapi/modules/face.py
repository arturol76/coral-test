import cvlib as cv
import cv2
import numpy as np
import os

import modules.detectors as detectors_model

import logging
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.name = "face"
        logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        return
        
    def get_model_name(self):
        return self.name

    def detect(
            self,
            image_cv
        ) -> detectors_model.DetectorResponse:

        #TO BE MODIFIED ----------------------
        gender = False
        #TO BE MODIFIED ----------------------

        faces_bbox, faces_conf = cv.detect_face(image_cv)
        
        bbox=[]
        label=[]
        conf=[]
        
        for bbox_item, conf_item in zip(faces_bbox, faces_conf):
            startX = bbox_item[0]
            startY = bbox_item[1]
            endX = bbox_item[2]
            endY = bbox_item[3]

            c = float(conf_item)
            l = 'person'
            b = [int(startX), int(startY), int(endX), int(endY)]

            bbox.append(b)
            label.append(l)
            conf.append(c)
        
        # for faces_item, conf_item in zip(faces, conf):
        #     logger.debug("type={}, confidence={:.2f}%".format(faces_item,conf_item))
            
        #     (startX, startY) = faces_item[0], faces_item[1]
        #     (endX, endY) = faces_item[2], faces_item[3]

        #     c = float(conf_item)
        #     l = 'face'
        #     b = [int(startX), int(startY), int(endX), int(endY)]

        #     bbox.append(b)
        #     label.append(l)
        #     conf.append(c)
            
        #     if gender == True:
        #         face_crop = np.copy(image_cv[startY:endY, startX:endX])
        #         (gender_label_arr, gender_confidence_arr) = cv.detect_gender(face_crop)
        #         idx = np.argmax(gender_confidence_arr)
                
        #         gender_label = gender_label_arr[idx]
        #         gender_confidence = "{:.2f}%".format(gender_confidence_arr[idx] * 100)
                
        #         #obj['gender'] = gender_label
        #         #obj['gender_confidence'] = gender_confidence

        model_response = detectors_model.DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c,self.get_model_name())

        return model_response
