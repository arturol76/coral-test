import cvlib as cv
import cv2
import numpy as np
import modules.globals as g
import os

import logging
logger = logging.getLogger(__name__)

class Detector:
    def __init__(self):
        self.name = "face"
        logger.debug('Initialized detector: {}'.format(self.name))

    def init(self):
        return
        
    def get_name(self):
        return self.name

    def detect(self, fi, fo, args):
        logger.debug("Reading {}".format(fi))
        image = cv2.imread(fi)
        faces, conf = cv.detect_face(image)

        logger.debug("faces={}".format(faces))
        logger.debug("conf={}".format(conf))

        detections = []
        for f, c in zip(faces, conf):
            c = "{:.2f}%".format(c * 100)

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            cv2.rectangle(image, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)
            rect = [int(startX), int(startY), int(endX), int(endY)]

            obj = {
                'type': 'face',
                'confidence': c,
                'box': rect
            }

            if args['gender']:
                face_crop = np.copy(image[startY:endY, startX:endX])
                (gender_label_arr, gender_confidence_arr) = cv.detect_gender(face_crop)
                idx = np.argmax(gender_confidence_arr)
                gender_label = gender_label_arr[idx]
                gender_confidence = "{:.2f}%".format(
                    gender_confidence_arr[idx] * 100)
                obj['gender'] = gender_label
                obj['gender_confidence'] = gender_confidence
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(image, gender_label, (startX, Y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            logger.debug("{}".format(obj))
            detections.append(obj)

        if not args['delete']:
            logger.debug("Writing {}".format(fo))
            cv2.imwrite(fo, image)

        if args['delete']:
            logger.debug("Deleting file {}".format(fi))
            os.remove(fi)

        return detections
