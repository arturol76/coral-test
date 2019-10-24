import datetime
from enum import Enum
import modules.log as log
from typing import Dict

class Detectors:
    def __init__(self):
        self.detectors_dict = dict()
                
    def add(self, model: object):
        self.detectors_dict[model.get_name()] = model

    def detect(self, name: str, fi: str, fo: str, args: Dict[str, str]):
        if name in self.detectors_dict:
            return self.detectors_dict[name].detect(fi, fo, args)
        else:
            error = 'detector with name {} not found'.format(name)
            log.logging.info(error)
            raise Exception(error)

    def get(self):
        return self.detectors_dict.keys()
    
