#! /usr/bin/env python3
 
import cv2
import time
import argparse

from state_search import in_search_fn, out_search_fn
from state_detect import in_detect_fn, out_detect_fn
from state_id import in_id_fn, out_id_fn
from state_track import in_track_fn, out_track_fn

from utils import Flow


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=None, help='Path to model to import')


if __name__ == "__main__":
    args = parser.parse_args()            
    flow=Flow(args.model)
    flow.in_pos()

    
    while True:
        if flow.gui.ABORT: break
            
        if flow.is_search():
            in_search_fn(flow)
            out_search_fn(flow)
            pass
    
        elif flow.is_detect():
            in_detect_fn(flow)
            out_detect_fn(flow)
            pass
    
        elif flow.is_id():
            in_id_fn(flow)
            out_id_fn(flow)
            pass
    
        elif flow.is_track():
            in_track_fn(flow)
            out_track_fn(flow)
            pass

    print('finishing up...')
    cv2.destroyAllWindows()
    flow.camera.ptz.home()
    time.sleep(2)
    flow.camera.stop()
    print('done.')

        
    
