import os
import requests as py_requests
import uuid
from mimetypes import guess_extension

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from PIL import Image
import io
import typing

import logging
logger = logging.getLogger(__name__)
logger.debug("boh")

ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])

def file_ext(filename):
    f,e = os.path.splitext(filename)
    logger.debug('input={} --> f={} e={}'.format(filename,f,e))
    return e.lower()

# Checks if filename is allowed
def allowed_ext(ext):
    return ext.lower() in ALLOWED_EXTENSIONS

def get_file_from_url(url: str, upload_folder: str):
    
    unique_filename = str(uuid.uuid4())
    file_with_path_no_ext = os.path.join(upload_folder, unique_filename)
    ext = None
   
    logger.debug('Got url:{}'.format(url))
    ext = file_ext(url)
    r = py_requests.get(url, allow_redirects=True)
    
    cd = r.headers.get('content-disposition')
    ct = r.headers.get('content-type')
    
    if cd:
        cd = cd.replace('"','')
        ext = file_ext(cd)
        logger.debug('CD: extension {} derived from {}'.format(ext,cd))
    elif ct:
        ext = guess_extension(ct.partition(';')[0].strip())
        if ext == '.jpe': 
            ext = '.jpg'
        logger.debug('CT: extension {} derived from {}'.format(ext,ct))
        if not allowed_ext(ext):
            #abort(400, msg='filetype {} not allowed'.format(ext))   
            return     
    else:
        ext = '.jpg'
    
    logger.debug('saving: {}{}'.format(file_with_path_no_ext,ext))
    open(file_with_path_no_ext+ext, 'wb').write(r.content)
    
    logger.debug('get_file returned: {}{}'.format(file_with_path_no_ext,ext))
    return file_with_path_no_ext, ext


#extracts file from form, saves it to a file and returns the filename
def get_file_from_form(file: UploadFile, upload_folder: str):
    filename_no_ext = os.path.splitext(file.filename)[0]
    ext = os.path.splitext(file.filename)[1]
    file_with_path_no_ext = os.path.join(upload_folder, filename_no_ext)
    file_with_path = os.path.join(upload_folder, file.filename)
    
    #writes file
    logger.debug('saving pic into file {}'.format(file_with_path))
    contents = file.file.read()
    image_pil = Image.open(io.BytesIO(contents))
    image_pil.save(file_with_path)
    
    logger.debug('get_file returned: {}{}'.format(file_with_path_no_ext,ext))
    return file_with_path_no_ext, ext