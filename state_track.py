"Tracker state as part of the PTZ tracker finite state machine"
import time
import numpy as np
import cv2

from state_id import async_id


def in_track_fn(system):
    print('=== tracking')

    # start tracker on identified frame
    success = system.start_tracker()

    # set timer
    local_timer = time.time()
    timeout = 0.5

    # estimated moving average parameters
    estimated_drone_prob = 1.0
    alpha = 0.1

    while success:
        zoom = system.get_telemetry()[2]
        move(system, zoom)

        # get next frame
        frame = system.get_frame()
        if frame is None:
            continue

        # update tracker
        success, system.drone_bbox = system.tracker.update(frame)
        if not success:
            system.camera.ptz.stop()
            system.camera.ptz.zoomto(0)
            break

        # Draw bounding box
        x, y, w, h = system.drone_bbox
        cv_im = frame.copy()
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        cv2.rectangle(cv_im, p1, p2, (255, 0, 0), 2, 1)
        system.update_gui(frame=cv_im)

        # if time is left, identify object in bbox
        if time.time() - local_timer > timeout:
            status = async_id(system, frame)

            # exponential weighted moving average
            estimated_drone_prob = (
                1 - alpha) * estimated_drone_prob + alpha * status

            # reset timer
            local_timer = time.time()

            # drone is probably lost
            if estimated_drone_prob < 0.5:
                break

        # user quit inititated
        if system.gui.RESET or system.gui.ABORT:
            system.gui.RESET = False
            break

    system.fsm.lost_track()


def move(system, zoom):

    # control camera
    x, y, w, h = system.drone_bbox
    center = (x + w / 2, y + h / 2)
    pan_error, tilt_error = system.camera.errors_pt(center,
                                                    system.camera.width,
                                                    system.camera.height)
    zoom_error = system.camera.error_zoom(w, system.camera.height)
    system.camera.control(pan_error=pan_error / scale_factor(zoom),
                          tilt_error=tilt_error / scale_factor(zoom))
    system.camera.control_zoom(zoom_error)


def scale_factor(zoom_value):
    factor = 0.1214 * np.exp(318.16 * 10**(-6) * zoom_value) + 1.0605
    return factor * 1.3


def out_track_fn(system):
    print('lost_track')

    # reinitialize short-term tracker
    system.tracker = cv2.TrackerCSRT_create()
