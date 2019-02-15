#! /usr/bin/env python3
 

'''
Finite state machine for ptz aerial tracking

 /''''''''''''''''''\
|       search       | <--------------.
 \__________________/                 |
   /|\          |                     |
    |timeout    |in_pos               |
    |          \|/                    |
 /''''''''''''''''''\                 |
|       detect       |                |
 \__________________/                 |
   /|\          |                     |
    |not_drone  |found_obj            |
    |          \|/                    |
 /''''''''''''''''''\                 |
|         id         |                |
 \__________________/                 |
          |                           |
          |drone                      |
         \|/                          |
 /''''''''''''''''''\  lost_track     |
|       track        |----------------'
 \__________________/
        

pip install transitions
'''


import time
import random
from transitions import Machine, State
from threading import Timer

from state_search import in_search_fn, out_search_fn
from state_detect import in_detect_fn, out_detect_fn
from state_id import in_id_fn, out_id_fn
from state_track import in_track_fn, out_track_fn
from utils import *

class Flow(object):
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
    
    def __init__(self):
        self.timer_expir = True # bool for if timer expired
        self.obj_detected = False # bool for if object detected
        self.machine = Machine(self, states=Flow.states, transitions=Flow.transitions, initial='search', auto_transitions=False)
        self.camera = Camera(PIDController(),PIDController(),PIDController())
        self.bg_model = None
        self.cur_imgs = None
        self.cur_bboxes = None
        self.drone_bbox = None
        
    def expiry(self):
        self.timer_expir = True

###########################################################
        
    def on_enter_search(self):
        in_search_fn(self)        
        
    def on_exit_search(self):
        out_search_fn(self)
        
    def on_enter_detect(self):
        in_detect_fn(self)

    def on_exit_detect(self):
        out_detect_fn(self)

    def on_enter_id(self):
        in_id_fn(self)
        
    def on_exit_id(self):
        out_id_fn(self)

    def on_enter_track(self):
        in_track_fn(self)
        
    def on_exit_track(self):
        out_track_fn(self)

 
flow=Flow()

flow.in_pos()
