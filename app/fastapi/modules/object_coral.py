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

class ObjectCoral:
    def __init__(self):
        log.logger.debug('Initialized Object Detection: CORAL')
        
        # Initialize engine
        self.model_file="/usr/share/edgetpu/examples/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
        self.label_file="/usr/share/edgetpu/examples/models/coco_labels.txt"
        self.engine = DetectionEngine(self.model_file)
        self.labels = self.ReadLabelFile(self.label_file) if self.label_file else None

    # Function to read labels from text files.
    def ReadLabelFile(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    # runs yolov3 object detection
    def detect(self, f,ext, args):
        fi = f+ext
        fo = f+'-object'+ext

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
                if self.labels:
                    log.logger.debug(self.labels[obj.label_id])
                #log.logger.debug ("score={}".format(obj.score))
                box = obj.bounding_box.flatten().tolist()
                #log.logger.debug ("box={}".format(box))
                                
                #sample output
                #score =  0.97265625
                #box =  [417.078184068203, 436.7141185646848, 2443.3632068037987, 1612.3385782686541]
                box = obj.bounding_box.flatten().tolist()
                box = [int(i) for i in box] #convert to int
                obj = {
                    'type': self.labels[obj.label_id],
                    'confidence': "{:.2f}%".format(obj.score * 100),
                    'box': box
                }
                log.logger.debug("{}".format(obj))
                detections.append(obj)
            #img.save(fo)
        else:
            log.logger.debug('No object detected!')

        return detections
