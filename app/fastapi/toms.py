#!/usr/bin/python3
import argparse
import uvicorn
import os
import modules.log as log

from pydantic import BaseModel
from typing import List, Set, Tuple, Dict

import toml

class API_DetectRequest(BaseModel):
    url: str = None
    file: str = None
    models_list: List[str] = None
    delete: bool = None

obj = API_DetectRequest()

obj.url = "art"
obj.models_list = ["a","b"]

print(toml.dumps(obj))