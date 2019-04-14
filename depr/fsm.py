
import cv2
import time
import random
from transitions import Machine, State
from threading import Timer

from state_search import in_search_fn, out_search_fn 
from state_detect import in_detect_fn, out_detect_fn
from state_id import in_id_fn, out_id_fn
from state_track import in_track_fn, out_track_fn
from utils import *             
from dnn import initialize_net

class Flow(object):
    '''
    Finite state machine for ptz aerial tracking
    
 /''''''''''''''''''\
|       search       | <--------------.
 \__________________/                 |
    /|\          |                    |
     |timeout    |in_pos              |
     |          \|/                   |
 /''''''''''''''''''\                 |
|       detect       |                |
 \__________________/                 |
    /|\          |                    |
     |not_drone  |found_obj           |
     |          \|/                   |
 /''''''''''''''''''\                 |
|         id         |                |
 \__________________/                 |
     |                                |
     |drone                           |
    \|/                               |
 /''''''''''''''''''\  lost_track     |
|       track        |----------------'
 \__________________/
        

pip install transitions
'''
    states=[
        State('search'),
        State('detect'),
        State('id'),
        State('track')
    ]
    transitions = [
    { 'trigger': 'in_pos', 'source': 'search', 'dest': 'detect' },
    { 'trigger': 'timeout', 'source': 'detect', 'dest': 'search' },
    { 'trigger': 'found_obj', 'source': 'detect', 'dest': 'id' },
    { 'trigger': 'drone', 'source': 'id', 'dest': 'track' },
    { 'trigger': 'not_drone', 'source': 'id', 'dest': 'detect' },
    { 'trigger': 'lost_track', 'source': 'track', 'dest': 'search' }
    ]
    
    def __init__(self, model_path):

        # Variables
        self.timer_expir = True # bool for if timer expired
        self.obj_detected = False # bool for if object detected
        self.cur_imgs = []
        self.cur_bboxes = []
        self.drone_bbox = None
        self.bg_model = None
        self.timeout_interval = 5

        # Objects
        self.machine = Machine(self, 
                               states=Flow.states, 
                               transitions=Flow.transitions, 
                               initial='search', 
                               auto_transitions=False)
        self.camera = Camera(PIDController(),PIDController(),PIDController())
        self.tracker = cv2.TrackerCSRT_create()
        self.timer_obj = Timer(self.timeout_interval, self.expiry, ())
        self.network = initialize_net(model_path)
        
        # Initialization routine
        init_count = 0
        while init_count < 5:
            _ = self.camera.cvreader.Read()
            init_count += 1

            
    def expiry(self):
        self.timer_expir = True    
