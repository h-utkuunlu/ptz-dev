import argparse
#import imutils
import time
import cv2

def in_track_fn(parent):
    print('=== tracking')

    cv2.namedWindow("gui",cv2.WINDOW_NORMAL)
    frame = parent.camera.cvreader.Read()
    success = parent.tracker.init(frame, parent.drone_bbox)

    while success:
        move(parent)
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
        cv2.imshow("gui", cv_im)
        cv2.waitKey(1)
    parent.lost_track()

def move(parent):
    
    # control camera
    x, y, w, h = parent.drone_bbox
    center = (x + w/2, y + h/2)
    pan_error, tilt_error = parent.camera.errors_pt(center, parent.camera.width, parent.camera.height)
    zoom_error = parent.camera.error_zoom((w+h)/2, parent.camera.height)
    parent.camera.control(pan_error=pan_error, tilt_error=tilt_error)
    parent.camera.control_zoom(zoom_error)
    return (x,y,w,h)

def out_track_fn(parent):
    print('lost_track')
    #parent.tracker.clear()
    parent.tracker = cv2.TrackerCSRT_create()
    
