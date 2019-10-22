import os
import requests as py_requests
import uuid
from mimetypes import guess_extension

ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg'])

def file_ext(filename):
    f,e = os.path.splitext(filename)
    print('input={} --> f={} e={}'.format(filename,f,e))
    return e.lower()

# Checks if filename is allowed
def allowed_ext(ext):
    return ext.lower() in ALLOWED_EXTENSIONS

def get_file(args, upload_folder):
    unique_filename = str(uuid.uuid4())
    file_with_path_no_ext = os.path.join(upload_folder, unique_filename)
    ext = None

    print (args)
    # uploaded as multipart data
    if args['file']:
        file = args['file']
        ext = file_ext(file.filename)
        if file.filename and allowed_ext(ext):
            file.save(file_with_path_no_ext+ext)
        else:
            #abort (500, msg='Bad file type {}'.format(file.filename))
            return

    # passed as a payload url
    elif args['url']:
        url = args['url']
        print('Got url:{}'.format(url))
        ext = file_ext(url)
        r = py_requests.get(url, allow_redirects=True)
        
        cd = r.headers.get('content-disposition')
        ct = r.headers.get('content-type')
        if cd:
            cd = cd.replace('"','')
            ext = file_ext(cd)
            print('CD: extension {} derived from {}'.format(ext,cd))
        elif ct:
            ext = guess_extension(ct.partition(';')[0].strip())
            if ext == '.jpe': 
                ext = '.jpg'
            print('CT: extension {} derived from {}'.format(ext,ct))
            if not allowed_ext(ext):
                #abort(400, msg='filetype {} not allowed'.format(ext))   
                return     
        else:
            ext = '.jpg'
        print('saving: {}{}'.format(file_with_path_no_ext,ext))
        open(file_with_path_no_ext+ext, 'wb').write(r.content)
    else:
        #abort(400, msg='could not determine file type')
        return

    print('get_file returned: {}{}'.format(file_with_path_no_ext,ext))
    return file_with_path_no_ext, ext