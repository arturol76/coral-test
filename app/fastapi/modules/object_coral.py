import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import numpy as np
import os

from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from PIL import ImageDraw

import modules.log as log

conf_min = 0.0

class Detector:
    def __init__(self):
        self.name = "object_coral"
        log.logger.debug('Initialized detector: {}'.format(self.name))
        
        # Initialize engine
        self.model_file="/usr/share/edgetpu/examples/models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
        self.label_file="/usr/share/edgetpu/examples/models/coco_labels.txt"
        self.engine = DetectionEngine(self.model_file)
        self.labels = self.ReadLabelFile(self.label_file) if self.label_file else None
        
    def get_name(self):
        return self.name
		
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
    def detect(self, fi, fo, args):
        
        # Open image.
        pil_image = Image.open(fi)
        log.logger.debug("Reading {}".format(fi))

        # Run inference.
        ans = self.engine.detect_with_image(pil_image, threshold=0.05, keep_aspect_ratio=True,relative_coord=False, top_k=10)
        
        detections = []
        post_bbox = []
        post_label = []
        post_conf = []

        # Display result.
        if ans:
            for obj in ans:
                log.logger.debug ('-----------------------------------------')
                if self.labels:
                    #sample output
                    #score =  0.97265625
                    #box =  [417.078184068203, 436.7141185646848, 2443.3632068037987, 1612.3385782686541]
                    box = obj.bounding_box.flatten().tolist()
                    box = [int(i) for i in box] #convert to int
                    
                    l = self.labels[obj.label_id]
                    c = obj.score
                    b = box
                    
                    obj = {
                            'type': l,
                            'confidence': "{:.2f}%".format(c * 100),
                            'box': b
                        }
                    log.logger.debug("{}".format(obj))

                    if c > conf_min:
                        detections.append(obj)
                        post_bbox.append(b)
                        post_label.append(l)
                        post_conf.append(c)
                    else:
                        log.logger.debug("DISCARDED as conf={} and threashold={}".format(c, conf_min))

            # Perform non maximum suppression to eliminate redundant overlapping boxes with
            # lower confidences.

            nms_box = []
            nms_conf = []
            for i in post_bbox:
                x = i[0]
                y = i[1]
                w = i[2] - i[0]
                h = i[3] - i[1]
                nms_box.append([x,y,w,h])

            for j in post_conf:
                nms_conf.append(float(j))

            log.logger.debug("NMS bbox {}".format(nms_box))
            log.logger.debug("NMS conf {}".format(nms_conf))

            #la prima soglia rappresenta la confidenza minima che accetti per un bbox
            #ti pare di aver capito che la seconda soglia rappresenta
            #l'area in % di sovrapposizione tra due bbox
            indices = cv2.dnn.NMSBoxes(nms_box, nms_conf, 0.2, 0.3)
            log.logger.debug("NMS indices {}".format(indices))

            nms_bbox = []
            nms_label = []
            nms_conf = []

            for i in indices:
                j=i[0]
                nms_bbox.append(post_bbox[j])
                nms_label.append(post_label[j])
                nms_conf.append(post_conf[j])

            cv_image = np.array(pil_image)
            out = draw_bbox(cv_image, nms_bbox, nms_label, nms_conf, write_conf=True)
            #log.logger.debug("Writing {}".format(fo))
            img2 = Image.fromarray(out, 'RGB')
            img2.save(fo)

            # if not args['delete']:
            #     cv_image = np.array(pil_image)
            #     out = draw_bbox(cv_image, post_bbox, post_label, post_conf, write_conf=True)
            #     #log.logger.debug("Writing {}".format(fo))
            #     img2 = Image.fromarray(out, 'RGB')
            #     img2.save(fo)
        
        else:
            log.logger.debug('No object detected!')

        if args['delete']:
            log.logger.debug("Deleting file {}".format(fi))
            os.remove(fi)

        return detections
