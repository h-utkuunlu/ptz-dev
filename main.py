#! /usr/bin/env python3
"Starts the finite state machine for the PTZ tracker"
import time
import argparse
import cv2

from state_search import in_search_fn, out_search_fn
from state_detect import in_detect_fn, out_detect_fn
from state_id import in_id_fn, out_id_fn
from state_track import in_track_fn, out_track_fn
from utils import LTT

parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--model',
                    default=None,
                    help='Path to model to import')

if __name__ == "__main__":
    args = parser.parse_args()
    system = LTT(args.model)
    system.fsm.in_pos()

    while True:
        if system.gui.ABORT:
            break
        if system.fsm.is_search():
            in_search_fn(system)
            out_search_fn(system)
        elif system.fsm.is_detect():
            in_detect_fn(system)
            out_detect_fn(system)
        elif system.fsm.is_id():
            in_id_fn(system)
            out_id_fn(system)
        elif system.fsm.is_track():
            in_track_fn(system)
            out_track_fn(system)

    print('Exiting...')
    cv2.destroyAllWindows()
    system.camera.ptz.home()
    time.sleep(2)
    system.camera.stop()
    system.logger.close()
    print('\n\ndone.')
