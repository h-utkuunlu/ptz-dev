from threading import Timer
import time

def in_search_fn(parent):
    print('=== search')
    local_timer = time.time()
    parent.camera.ptz.goto(0,0,24)
    home = False
    while not home:
        if time.time() > local_timer + 0.5:
            if parent.camera.get_pan_tilt_position()[0] == 0:
                home=True
    parent.in_pos()

def out_search_fn(parent):
    parent.bg = None # bg.generate()
    parent.cur_imgs = None
    parent.cur_bboxes = None
    parent.timer_expir = False
    Timer(5, parent.expiry, ()).start()
