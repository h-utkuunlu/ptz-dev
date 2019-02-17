import argparse
#import imutils
import time
import cv2

def in_track_fn(parent):
    print('=== tracking')
    move(parent)  # control camera based on drone bbox
    frame = parent.camera.cvreader.Read()
    success = parent.tracker.init(frame, parent.drone_bbox)
    while success:
        frame = parent.camera.cvreader.Read()
        if frame is None:
            continue
        success, parent.drone_bbox = parent.tracker.update(frame)
        if not success:
            break

        move(parent)
        # Draw bounding box
        cv_im = frame.clone()
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(cv_im, p1, p2, (255,0,0), 2, 1)
        cv2.imshow("gui", cv_im)
    parent.lost_track()

def move(parent):
    x, y, w, h = parent.drone_bbox[0], parent.drone_bbox[1], parent.drone_bbox[2], parent.drone_bbox[3]
    # control camera
    center = (x + w/2, y + h/2)
    pan_error, tilt_error = parent.camera.errors_pt(center, parent.camera.width, parent.camera.height)
    zoom_error = parent.camera.error_zoom((w+h)/2, parent.camera.height)
    parent.camera.control(pan_error=pan_error, tilt_error=tilt_error)
    parent.camera.control_zoom(zoom_error)

def out_track_fn(parent):
    print('lost_track')
    parent.tracker.clear()

