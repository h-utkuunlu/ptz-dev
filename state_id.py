import time
import random

def in_id_fn(parent):
    print('id-ing object')
    time.sleep(1)
    if random.random() > 0.5:
        print('not drone :(')
        parent.not_drone()
    else:
        print('drone !')
        parent.drone()

def out_id_fn(parent):
    pass
