from edgetpu.detection.engine import DetectionEngine
from PIL import Image

from modules.detectors import DetectorResponse
from modules.DetectorBase import DetectorBase

import logging
logger = logging.getLogger(__name__)

class Detector(DetectorBase):
    def __init__(self):
        DetectorBase.__init__(self, "face_coral")
        
    def init(self):
        # Initialize engine
        self.model_file="/usr/share/edgetpu/examples/models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite"
        self.label_file=None
        
        try:
            self.engine = DetectionEngine(self.model_file)
            
        except Exception as error:
            logger.error('Initializion error: {}'.format(error))
            
        return
	
    # runs yolov3 object detection
    def detect(
            self,
            image_cv
        ) -> DetectorResponse:
        
        pil_image = Image.fromarray(image_cv) # convert opencv frame (with type()==numpy) into PIL Image
        
        # Run inference.
        ans = self.engine.detect_with_image(pil_image, threshold=0.05, keep_aspect_ratio=True,relative_coord=False, top_k=10)
        
        bbox = []
        conf = []
        label = []

        # Display result.
        if ans:
            for obj in ans:
                #sample output
                #score =  0.97265625
                #box =  [417.078184068203, 436.7141185646848, 2443.3632068037987, 1612.3385782686541]
                b = obj.bounding_box.flatten().tolist()
                b = [int(i) for i in b] #convert to int
                c = float(obj.score)
                l = 'person'
                                
                bbox.append(b)
                conf.append(c)
                label.append(l)

        else:
            logger.debug('No face detected!')

        model_response = DetectorResponse(self.get_model_name())
        for l, c, b in zip(label, conf, bbox):
            model_response.add(b,l,c,self.get_model_name())

        return model_response
