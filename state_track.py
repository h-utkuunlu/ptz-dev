import argparse
#import imutils
import time
import cv2
from state_id import async_id
import numpy as np
def in_track_fn(parent):
    print('=== tracking')

    frame = parent.gui.frame
    success = parent.tracker.init(frame, parent.drone_bbox)

    local_timer = time.time()
    timeout = 0.5
    
    while success:
        zoom = parent.camera.cvreader.ReadTelemetry()[2]
        move(parent, zoom)
        frame = parent.camera.cvreader.Read()
        if frame is None:
            continue
        success, parent.drone_bbox = parent.tracker.update(frame)
        if not success:
            parent.camera.ptz.stop()
            parent.camera.ptz.zoomto(0)
            break

        # Draw bounding box
        x, y, w, h = parent.drone_bbox
        cv_im = frame.copy()
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(cv_im, p1, p2, (255,0,0), 2, 1)
        parent.gui.update(frame=cv_im)
        parent.logger.vout.write(cv_im) # logger video out
        if time.time() - local_timer > timeout:
            local_timer = time.time()
            status = async_id(parent)
            if status != 1:
                break
        if parent.gui.RESET or parent.gui.ABORT:
            parent.gui.RESET = False
            break

        #Log
        parent.logger.log()
    
    parent.lost_track()

def move(parent, zoom):
    
    # control camera
    x, y, w, h = parent.drone_bbox
    center = (x + w/2, y + h/2)
    pan_error, tilt_error = parent.camera.errors_pt(center, parent.camera.width, parent.camera.height)
    zoom_error = parent.camera.error_zoom(max(w, h), parent.camera.height)
    parent.camera.control(pan_error=pan_error/scale_factor(zoom), tilt_error=tilt_error/scale_factor(zoom))
    parent.camera.control_zoom(zoom_error)
    return (x,y,w,h)
    
def scale_factor(zoom_factor):
    zoom_value = (zoom_factor-1)*862.32
    factor = 0.1214*np.exp(318.16*10**(-6)*zoom_value) + 1.0605
    return factor

def out_track_fn(parent):
    print('lost_track')
    #parent.tracker.clear()
    parent.tracker = cv2.TrackerCSRT_create()
    
