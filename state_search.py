from threading import Timer
import time
def in_search_fn(parent):
    print('searching')
    time.sleep(1)
    parent.in_pos()

def out_search_fn(parent):
    print('bg_initialized')
    parent.timer_expir = False
    Timer(5, parent.expiry, ()).start()
