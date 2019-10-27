import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import modules.globals as g
import os

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw

import modules.log as log

class Detector:
    def __init__(self):
        self.name = "face_coral"
        log.logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        # Initialize engine
        self.model_file="/usr/share/edgetpu/examples/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite"
        self.label_file=None
        
        try:
            self.engine = DetectionEngine(self.model_file)
            self.labels = self.ReadLabelFile(self.label_file) if self.label_file else None

        except Exception as error:
            log.logger.error('Initializion error: {}'.format(error))
            
        return

    def get_name(self):
        return self.name
		
    # runs yolov3 object detection
    def detect(self, fi, fo, args):
        
        # Open image.
        img = Image.open(fi)
        log.logger.debug("Reading {}".format(fi))

        # Run inference.
        ans = self.engine.detect_with_image(img, threshold=0.05, keep_aspect_ratio=True,relative_coord=False, top_k=10)
        
        detections = []

        # Display result.
        if ans:
            for obj in ans:
                log.logger.debug ('-----------------------------------------')
                
                #sample output
                #score =  0.97265625
                #box =  [417.078184068203, 436.7141185646848, 2443.3632068037987, 1612.3385782686541]
                box = obj.bounding_box.flatten().tolist()
                box = [int(i) for i in box] #convert to int
                obj = {
                    'confidence': "{:.2f}%".format(obj.score * 100),
                    'box': box
                }
                log.logger.debug("{}".format(obj))
                detections.append(obj)
            
            img.save(fo)
        else:
            log.logger.debug('No face detected!')

        if args['delete']:
            log.logger.debug("Deleting file {}".format(fi))
            os.remove(fi)

        return detections
