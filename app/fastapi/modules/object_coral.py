import cv2
import os

from edgetpu.detection.engine import DetectionEngine
from PIL import Image

from modules.DetectorResponse import DetectorResponse
from modules.DetectorBase import DetectorBase

import logging
logger = logging.getLogger(__name__)

conf_min = 0.0

class Detector(DetectorBase):
    def __init__(self):
        DetectorBase.__init__(self, "object_coral")
        
    def init(self):
        # Initialize engine
        self.model_file="/usr/share/edgetpu/examples/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
        self.label_file="/usr/share/edgetpu/examples/models/coco_labels.txt"

        try:
            self.engine = DetectionEngine(self.model_file)
            self.labels = self.__read_label_file(self.label_file) if self.label_file else None

        except Exception as error:
            logger.error('Initializion error: {}'.format(error))
            
        return
        
    # Function to read labels from text files.
    def __read_label_file(self, file_path):
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        ret = {}
        for line in lines:
            pair = line.strip().split(maxsplit=1)
            ret[int(pair[0])] = pair[1].strip()
        return ret

    # runs yolov3 object detection
    def detect(
            self, 
            image_cv
        ) -> DetectorResponse:
        
        pil_image = Image.fromarray(image_cv) # convert opencv frame (with type()==numpy) into PIL Image
        
        # Run inference.
        ans = self.engine.detect_with_image(pil_image, threshold=0.05, keep_aspect_ratio=True,relative_coord=False, top_k=10)
        
        pre_bbox = []
        pre_label = []
        pre_conf = []

        # Display result.
        if ans:
            for obj in ans:
                if self.labels:
                    #sample output
                    #score =  0.97265625
                    #box =  [417.078184068203, 436.7141185646848, 2443.3632068037987, 1612.3385782686541]
                    box = obj.bounding_box.flatten().tolist()
                    box = [int(i) for i in box] #convert to int
                    
                    l = self.labels[obj.label_id]
                    c = float(obj.score)
                    b = box

                    # x = int(box[0])
                    # y = int(box[1])
                    # w = int(box[2] - box[0])
                    # h = int(box[3] - box[1])
                    # b = [x,y,w,h]

                    pre_bbox.append(b)
                    pre_label.append(l)
                    pre_conf.append(c)
                    
                    # if c > conf_min:
                    #     post_bbox.append(b)
                    #     post_label.append(l)
                    #     post_conf.append(c)
                    # else:
                    #     logger.debug("DISCARDED as conf={} and threashold={}".format(c, conf_min))

            # Perform non maximum suppression to eliminate redundant overlapping boxes with
            # lower confidences.

            #la prima soglia rappresenta la confidenza minima che accetti per un bbox
            #ti pare di aver capito che la seconda soglia rappresenta
            #l'area in % di sovrapposizione tra due bbox
            indices = cv2.dnn.NMSBoxes(pre_bbox, pre_conf, 0.2, 0.3)
            logger.debug("NMS indices {}".format(indices))

            bbox = []
            label = []
            conf = []

            for i in indices:
                j=i[0]
                bbox.append(pre_bbox[j])
                label.append(pre_label[j])
                conf.append(pre_conf[j])
        else:
            logger.debug('No object detected!')

        model_response = DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c,self.get_model_name())

        return model_response  
