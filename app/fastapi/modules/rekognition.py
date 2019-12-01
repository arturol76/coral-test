import os
import boto3
import io
from PIL import Image

from modules.DetectorResponse import DetectorResponse
from modules.DetectorBase import DetectorBase

import logging
logger = logging.getLogger(__name__)

class Detector(DetectorBase):
    def __init__(self):
        DetectorBase.__init__(self, "rekognition")

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

	#person|car|motorbike|bus|truck
    def __convert_label(self, label):
        return self.rekognition_labels[label]

    def detect(
            self,
            image_cv
        ) -> DetectorResponse:
        
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

                    l = self.__convert_label(reko_label['Name'])
                    c = float(reko_label['Confidence']/100)
                    b = [x1, y1, x2, y2]
                    
                    bbox.append(b)
                    label.append(l)
                    conf.append(c)

        model_response = DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c,self.get_model_name())

        return model_response
