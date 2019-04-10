from threading import Timer
import time
import random
import cv2
import numpy as np


def in_search_fn(parent):
    print('=== search')
    local_timer = time.time()
    #pan = int(random.random()*1000)+1
    pan = 0
    if pan > 500:
        pan = int((pan-500)*4.894/4)
    else:
        pan = int(65535 - pan*4.894/4)
    
    in_pos = False
    while not in_pos:
        if time.time() > local_timer + 0.5:
            local_timer = time.time()
            parent.telemetry = parent.camera.cvreader.ReadTelemetry()
            if parent.telemetry[0] == pan and parent.telemetry[2] == 1:
                in_pos = True
            else:
                parent.camera.ptz.goto(pan,0,24)
                parent.camera.ptz.zoomto(0)
                                
    print("in_pos")    
    parent.in_pos()

def out_search_fn(parent):
    pxcnt = 60
    parent.bg_model = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    init_count=0
    #cv2.namedWindow("bg model",cv2.WINDOW_NORMAL)
    while init_count < pxcnt+1:
        frame = parent.camera.cvreader.Read()
        #cv2.imshow('frame_orig',frame)
        if frame is None:
            continue
        if not parent.gui.initialized:
            parent.gui.init(frame)
        else:
            parent.gui.update(frame=frame)
        _ = parent.bg_model.apply(frame)
        parent.logger.vout.write(frame) # logger video out
        init_count += 1
    #bg_img = parent.bg_model.getBackgroundImage()
    #cv2.imshow("bg model", bg_img)
    #cv2.waitKey(1)
    print("bg_generated")
    parent.timer_expir = False
    parent.timer_obj = Timer(parent.timeout_interval, parent.expiry, ())  
    parent.timer_obj.start()
