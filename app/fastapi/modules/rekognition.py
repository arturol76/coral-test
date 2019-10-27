import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import modules.globals as g
import os

import modules.log as log

import boto3
import io
from PIL import Image

class Detector:
    def __init__(self):
        self.name = "rekognition"
        log.logger.debug('Initialized detector: {}'.format(self.name))

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

    def detect(self, fi, fo, args):
        log.logger.debug("Reading {}".format(fi))
        image = cv2.imread(fi)

        log.logger.debug("[REKOGNITION] request via boto3...")
        imgHeight, imgWidth = image.shape[:2]
        pil_img = Image.fromarray(image) # convert opencv frame (with type()==numpy) into PIL Image
        stream = io.BytesIO()
        pil_img.save(stream, format='JPEG') # convert PIL Image to Bytes
        bin_img = stream.getvalue()

        log.logger.debug("request via boto3...")
        response = self.client.detect_labels(Image={'Bytes': bin_img})
        #log.logger.debug("Reading {}".format(response))
        log.logger.debug("...response received")

        detections = []
        post_bbox = []
        post_label = []
        post_conf = []

        for label in response['Labels']:
            for instance in label['Instances']:
                if label['Name'] in self.rekognition_labels:
                    
                    #extracting bounding box coordinates
                    left = int(imgWidth * instance['BoundingBox']['Left'])
                    top = int(imgHeight * instance['BoundingBox']['Top'])
                    width = int(imgWidth * instance['BoundingBox']['Width'])
                    height = int(imgHeight * instance['BoundingBox']['Height'])

                    x1 = left
                    y1 = top
                    x2 = left+width
                    y2 = top+height

                    log.logger.debug ('-----------------------------------------')
                    l = self.convert_label(label['Name'])
                    c = "{:.2f}%".format(float(label['Confidence']))
                    b = [x1, y1, x2, y2]
                    log.logger.debug("box={}".format(b))
                    obj = {
                        'type': l,
                        'confidence': c,
                        'box': b
                    }
                    log.logger.debug("{}".format(obj))
                    detections.append(obj)

                    post_bbox.append(b)
                    post_label.append(l)
                    post_conf.append(c)

        if not args['delete']:
            out = draw_bbox(image, post_bbox, post_label, post_conf)
            log.logger.debug("Writing {}".format(fo))
            cv2.imwrite(fo, out)

        if args['delete']:
            log.logger.debug("Deleting file {}".format(fi))
            os.remove(fi)

        return detections
