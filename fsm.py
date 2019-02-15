#! /usr/bin/env python3

import time
import random
from transitions import Machine, State
from threading import Timer

class Flow(object):
	states=[
		State('obj_det'),
		State('search'),# on_exit=['on_exit_search']),
		State('obj_id'),
		State('obj_track')
	]
	transitions = [
	{ 'trigger': 'timeout', 'source': 'obj_det', 'dest': 'search' },
	{ 'trigger': 'found_obj', 'source': 'obj_det', 'dest': 'obj_id' },
	{ 'trigger': 'in_pos', 'source': 'search', 'dest': 'obj_det' },
	{ 'trigger': 'drone', 'source': 'obj_id', 'dest': 'obj_track' },
	{ 'trigger': 'not_drone', 'source': 'obj_id', 'dest': 'obj_det' },
	{ 'trigger': 'lost_track', 'source': 'obj_track', 'dest': 'search' }
]
	
	def __init__(self):
		self.timer_expir = True # bool for if timer expired
		self.obj_detected = False # bool for if object detected
		self.machine = Machine(self, states=Flow.states, transitions=Flow.transitions, initial='search', auto_transitions=False)

	def expiry(self):
		self.timer_expir = True

###########################################################
		
	def on_enter_search(self):
		print('searching')
		time.sleep(1)
		self.in_pos()
		
	def on_exit_search(self):
		print('bg initialized')
		self.timer_expir = False
		Timer(5, self.expiry, ()).start()

	def on_enter_obj_det(self):
		local_timer = time.time()
		while not self.timer_expir:
			if time.time()-local_timer>3:
				self.found_obj()
			print('detecting objects')
			time.sleep(1)
			
		self.timeout()

	def on_exit_obj_det(self):
		pass

	def on_enter_obj_id(self):
		print('id-ing object')
		time.sleep(1)
		if random.random() > 0.5:
			print('not drone :(')
			self.not_drone()
		else:
			print('drone !')
			self.drone()

	def on_exit_obj_id(self):
		pass

	def on_enter_obj_track(self):
		print('tracking')
		time.sleep(5)
		self.lost_track()
flow=Flow()

