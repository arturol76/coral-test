import numpy as np
import cv2
from enum import Enum
import datetime

from modules.DetectorResponse import DetectorResponse
from modules.DetectorBase import DetectorBase


import logging
logger = logging.getLogger(__name__)

class YoloModel(str, Enum):
    yolov3      = "yolov3"
    yolov3_tiny = "yolov3_tiny"
    yolov3_spp  = "yolov3_spp"

#https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.txt
#https://pjreddie.com/media/files/yolov3-tiny.weights
#https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg

yolov3_config_file="./models/yolov3.cfg"
yolov3_labels_file="./models/yolov3.txt"
yolov3_weights_file="./models/yolov3.weights"

yolov3_tiny_config_file="./models/yolov3-tiny.cfg"
yolov3_tiny_labels_file="./models/yolov3.txt"
yolov3_tiny_weights_file="./models/yolov3-tiny.weights"

yolov3_spp_config_file="./models/yolov3-spp.cfg"
yolov3_spp_labels_file="./models/yolov3.txt"
yolov3_spp_weights_file="./models/yolov3-spp.weights"

class Detector(DetectorBase):
    def __init__(self, model: YoloModel):
        DetectorBase.__init__(self, model)
        self.model = model

    def init(self):
        start = datetime.datetime.now()

        if self.model == YoloModel.yolov3_tiny:
            config_file_abs_path = yolov3_tiny_config_file
            weights_file_abs_path = yolov3_tiny_weights_file
            class_file_abs_path = yolov3_tiny_labels_file
        elif self.model == YoloModel.yolov3:
            config_file_abs_path = yolov3_config_file
            weights_file_abs_path = yolov3_weights_file
            class_file_abs_path = yolov3_labels_file
        else:
            config_file_abs_path = yolov3_spp_config_file
            weights_file_abs_path = yolov3_spp_weights_file
            class_file_abs_path = yolov3_spp_labels_file

        f = open(class_file_abs_path, 'r')
        self.classes = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)
        
        logger.debug('Initialized detector: {}'.format(self.name))
        logger.debug('config:{}, weights:{}'.format(config_file_abs_path, weights_file_abs_path))

        stop = datetime.datetime.now()
        elapsed_time = stop - start
        print("initialization took:", elapsed_time)
        
        return

    def get_classes(self):
        return self.classes

    def __get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def detect(
            self, 
            image_cv
        ) -> DetectorResponse:

        Height, Width = image_cv.shape[:2]
        scale = 0.00392
        
        blob = cv2.dnn.blobFromImage(image_cv, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.__get_output_layers())
        
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        bbox = []
        label = []
        conf = []
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            bbox.append( [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))])
            label.append(str(self.classes[class_ids[i]]))
            conf.append(float(confidences[i]))

        model_response = DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c,self.get_model_name())
            
        return model_response                                  

