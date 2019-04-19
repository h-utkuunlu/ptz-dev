"Search state as part of the PTZ tracker finite state machine"
from threading import Timer
import time
import cv2


def in_search_fn(system):
    """Search heuristic."""
    print('=== search')
    local_timer = time.time()
    #pan_target = int(random.random()*1000)+1
    pan_target = 0
    if pan_target > 500:
        pan_target = int((pan_target - 500) * 4.894 / 4)
    else:
        pan_target = int(65535 - pan_target * 4.894 / 4)

    # move camera into position
    in_pos = False
    while not in_pos:
        # Update GUI
        frame = system.get_frame()
        if frame is None:
            continue
        system.update_gui(frame=frame)

        time.sleep(1 / 30)
        if time.time() > local_timer + 0.5:
            local_timer = time.time()
            telemetry = system.get_telemetry()
            pan, tilt, zoom = telemetry
            if pan == pan_target and zoom == 0:
                in_pos = True
            else:
                system.camera.ptz.goto(pan_target, 0, 24)
                system.camera.ptz.zoomto(0)

    # trigger in_pos transition
    print("in_pos")
    system.fsm.in_pos()


def out_search_fn(system):
    """Generates background model."""
    system.bg_model = cv2.createBackgroundSubtractorKNN(detectShadows=False)

    pxcnt = 60
    init_count = 0
    while init_count < pxcnt:
        #for init_count in range(pxcnt):

        # get most recent frame
        frame = system.get_frame()
        if frame is None:
            continue
        system.update_gui(frame=frame)

        # create background
        system.bg_model.apply(frame)

        # increment bg generation counter
        init_count += 1

    print("bg_generated")

    # set and start timer for detection state
    system.timer_expir = False
    system.timer_obj = Timer(system.timeout_interval, system.expiry, ())
    system.timer_obj.start()
