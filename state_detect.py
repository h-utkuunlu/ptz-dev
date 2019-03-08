import time
import cv2
<<<<<<< HEAD
import numpy as np
=======
from utils import expand_bbox
>>>>>>> master

def add_padding(rect_par, ratio, dims):
    x, y, w, h = rect_par
    padx = int(w*ratio)
    pady = int(h*ratio)

    if x-padx < 0 or y-pady < 0 or x+w+2*padx > dims[0] or y+h+2*pady > dims[0]:
        return x, y, w, h
    else:
        return x-padx, y-pady, w+2*padx, h+2*pady

def in_detect_fn(parent):

    # Setup
    parent.cur_imgs = []
    parent.cur_bboxes = []
    
    # Tuning parameters
    kernel_size = 5
    padding_ratio = 0.1
    area_threshold = 450

    found = False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
    res = (parent.camera.width, parent.camera.height)

    # cv2.namedWindow("fgmask",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("main_window", parent.gui.mw_w*2, parent.gui.mw_h)
    # cv2.moveWindow("fgmask",parent.gui.mw_x+parent.gui.mw_w,parent.gui.mw_y)
    while not parent.timer_expir:
        frame = parent.camera.cvreader.Read()
        if frame is None:
            continue

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Make the grey scale image have three channels
        grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        numpy_horizontal = np.hstack((frame, grey_3_channel))

        numpy_horizontal_concat = np.concatenate((frame, grey_3_channel), axis=1)

        cv2.imshow('main_window', numpy_horizontal_concat)
        cv2.waitKey(1)
        # cv2.imshow("main_window",frame)
        # cv2.waitKey(1)
        fgmask = parent.bg_model.apply(frame)
<<<<<<< HEAD
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=3)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        #fgmask = cv2.erode(fgmask, kernel, iterations=8)
        # cv2.imshow("fgmask",fgmask)
        # cv2.waitKey(1)
=======
        fgmask = cv2.medianBlur(fgmask, 9)
        fgmask = cv2.dilate(fgmask, kernel, iterations=5)

        cv2.imshow("fgmask",fgmask)
        cv2.waitKey(1)
>>>>>>> master
        _, contours, _ = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        
        for c in contours:
            if cv2.contourArea(c) > area_threshold:
                found = True
                rect_par = cv2.boundingRect(c)
                x, y, w, h = add_padding(rect_par, padding_ratio, res)
                parent.cur_bboxes.append((x, y, w, h))

                x, y, w, h = expand_bbox(*add_padding(rect_par, padding_ratio, res))
                parent.cur_imgs.append(frame[y:y+h, x:x+w])
                #parent.cur_bboxes.append((x, y, w, h))
                
        if found:
            parent.found_obj()
            return
            
    parent.timeout()

def out_detect_fn(parent):
    pass


