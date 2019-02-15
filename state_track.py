import time

def in_track_fn(parent):
	print('tracking')
	time.sleep(5)
	parent.lost_track()

def out_track_fn(parent):
	pass
