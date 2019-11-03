import cvlib as cv
import cv2
import numpy as np
import modules.globals as g
import os

import boto3
import io
from PIL import Image

import logging
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.name = "rekognition"
        logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        self.aws_region = os.environ['AWS_REGION']
        self.aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
        self.aws_access_key_secret = os.environ['AWS_ACCESS_KEY_SECRET']
        
        self.client=boto3.client(
            'rekognition',
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_access_key_secret
        )

        self.rekognition_labels= {
            "Person": "person",
            "Car": "car"
        }
        return

    def get_name(self):
        return self.name
		
	#person|car|motorbike|bus|truck
    def convert_label(self, label):
        return self.rekognition_labels[label]

    def detect(
            self,
            image_cv
        ):
        
        logger.debug("[REKOGNITION] request via boto3...")
        imgHeight, imgWidth = image_cv.shape[:2]
        pil_img = Image.fromarray(image_cv) # convert opencv frame (with type()==numpy) into PIL Image
        stream = io.BytesIO()
        pil_img.save(stream, format='JPEG') # convert PIL Image to Bytes
        bin_img = stream.getvalue()

        logger.debug("request via boto3...")
        response = self.client.detect_labels(Image={'Bytes': bin_img})
        #logger.debug("Reading {}".format(response))
        logger.debug("...response received")

        bbox = []
        label = []
        conf = []

        for reko_label in response['Labels']:
            for instance in reko_label['Instances']:
                if reko_label['Name'] in self.rekognition_labels:
                    
                    #extracting bounding box coordinates
                    left = int(imgWidth * instance['BoundingBox']['Left'])
                    top = int(imgHeight * instance['BoundingBox']['Top'])
                    width = int(imgWidth * instance['BoundingBox']['Width'])
                    height = int(imgHeight * instance['BoundingBox']['Height'])

                    x1 = left
                    y1 = top
                    x2 = left+width
                    y2 = top+height

                    l = self.convert_label(reko_label['Name'])
                    c = float(reko_label['Confidence']/100)
                    b = [x1, y1, x2, y2]
                    
                    bbox.append(b)
                    label.append(l)
                    conf.append(c)

        for l, c, b in zip(label, conf, bbox):
            logger.debug("type={}, confidence={:.2f}%, box={}".format(l,c,b))

        return bbox, label, conf
