import os
import requests as py_requests
import uuid
from mimetypes import guess_extension

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from PIL import Image
import io
import typing

import cv2 #drawbox

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

def draw_bbox2(
        img, 
        bbox, 
        labels, 
        confidence, 
        color=None, 
        write_conf=True
    ):

   # g.logger.debug ("DRAW BBOX={} LAB={}".format(bbox,labels))
    slate_colors = [ 
            (39, 174, 96),
            (142, 68, 173),
            (0,129,254),
            (254,60,113),
            (243,134,48),
            (91,177,47)
        ]
    
    # if no color is specified, use my own slate
    if color is None:
            # opencv is BGR
        bgr_slate_colors = slate_colors[::-1]

    # first draw the polygons, if any
    newh, neww = img.shape[:2]
    
    # now draw object boundaries
    arr_len = len(bgr_slate_colors)
    for i, label in enumerate(labels):
        #=g.logger.debug ('drawing box for: {}'.format(label))
        color = bgr_slate_colors[i % arr_len]
        if write_conf and confidence:
            label += ' ' + str(format(confidence[i] * 100, '.2f')) + '%'
        # draw bounding box around object
        
        #g.logger.debug ("DRAWING RECT={},{} {},{}".format(bbox[i][0], bbox[i][1],bbox[i][2], bbox[i][3]))
        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)

        # write text 
        font_scale = 0.8
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 1
        #cv2.getTextSize(text, font, font_scale, thickness)
        text_size = cv2.getTextSize(label, font_type, font_scale , font_thickness)[0]
        text_width_padded = text_size[0] + 4
        text_height_padded = text_size[1] + 4

        r_top_left = (bbox[i][0], bbox[i][1] - text_height_padded)
        r_bottom_right = (bbox[i][0] + text_width_padded, bbox[i][1])
        cv2.rectangle(img, r_top_left, r_bottom_right, color, -1)
        #cv2.putText(image, text, (x, y), font, font_scale, color, thickness) 
        # location of text is botom left
        cv2.putText(img, label, (bbox[i][0] + 2, bbox[i][1] - 2), font_type, font_scale, [255, 255, 255], font_thickness)

    return img