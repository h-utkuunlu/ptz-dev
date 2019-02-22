from threading import Timer
import time
import random
import cv2

def in_search_fn(parent):
    print('=== search')
    local_timer = time.time()
    pan = int(random.random()*1000)+1
    if pan > 500:
        pan = int((pan-500)*4.894/4)
    else:
        pan = int(65535 - pan*4.894/4)
    parent.camera.ptz.goto(pan,0,24)
    home = False
    while not home:
        if time.time() > local_timer + 0.5:
            local_timer = time.time()
            telemetry = parent.camera.cvreader.ReadTelemetry()
            if telemetry[0] == pan and telemetry[2] == 1:
                home=True
            else:
                parent.camera.ptz.goto(pan,0,24)
                parent.camera.ptz.zoomto(0)
                                
    print("in_pos")    
    parent.in_pos()

def out_search_fn(parent):
    pxcnt=60
    #parent.bg_model = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=pxcnt)
    parent.bg_model = cv2.bgsegm.createBackgroundSubtractorMOG(history=pxcnt*5, nmixtures=5)
    init_count=0
    #cv2.namedWindow("bg model",cv2.WINDOW_NORMAL)
    while init_count < pxcnt+1:
        frame = parent.camera.cvreader.Read()
        if frame is None:
            continue
        cv2.imshow("main_window",frame)
        cv2.waitKey(1)
        _ = parent.bg_model.apply(frame)
        
        init_count += 1
    #bg_img = parent.bg_model.getBackgroundImage()
    #cv2.imshow("bg model", bg_img)
    #cv2.waitKey(1)
    print("bg_generated")
    parent.timer_expir = False
    parent.timer_obj = Timer(parent.timeout_interval, parent.expiry, ())  
    parent.timer_obj.start()
