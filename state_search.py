from threading import Timer
import time
import random

def in_search_fn(parent):
    print('=== search')
    local_timer = time.time()
    pan = int(random.random()*1000)+1
    if pan > 500:
        pan = int((pan-500)*4.894)
    else:
        pan = int(65535 - pan*4.894)
    parent.camera.ptz.goto(pan,0,24)
    home = False
    while not home:
        if time.time() > local_timer + 0.5:
            if parent.camera.ptz.get_pan_tilt_position()[0] == pan:
                home=True
    parent.in_pos()

def out_search_fn(parent):
    pxcnt=20
    parent.bg_model = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=pxcnt)
    init_count=0
    while init_count < pxcnt:
        _ = parent.bg_model.apply(frame)
        init_count += 1
    parent.timer_expir = False
    parent.timer_obj = Timer(parent.timeout_interval, parent.expiry, ())
    parent.timer_obj.start()
