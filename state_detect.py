import time

def in_detect_fn(parent):
	local_timer = time.time()
	while not parent.timer_expir:
		if time.time()-local_timer>3:
			parent.found_obj()
		print('detecting objects')
		time.sleep(1)
	parent.timeout()

def out_detect_fn(parent):
	pass
