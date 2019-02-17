import time
import cv2

def add_padding(rect_par, ratio, dims):
    x, y, w, h = rect_par
    padx = int(w*ratio)
    pady = int(h*ratio)

    if x-padx < 0 or y-pady < 0 or x+w+2*padx > dims[0] or y+h+2*pady > dims[0]:
        return (x, y, w, h)
    else:
        return (x-padx, y-pady, w+2*padx, h+2*pady)

def in_detect_fn(parent):

    print("=== detect")

    # Setup
    parent.cur_imgs = []
    parent.cur_bboxes = []
    
    # Tuning parameters
    kernel_size = 5
    padding_ratio = 0.2
    area_threshold = 100

    found = False
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
    res = (parent.camera.width, parent.camera.height)
    # bug001: detects objects instantly even if camera is not connected -> results in an infinite loop of detect-id-detect that overflows trigger recursion depth of transitions
    while not parent.timer_expir:
        frame = parent.camera.cvreader.Read()
        if frame is None:
            continue
        fgmask = parent.bg_model.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fgmask = cv2.dilate(fgmask, kernel, iterations=10)
        fgmask = cv2.erode(fgmask, kernel, iterations=8)
        _, contours, _ = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
        
        for c in contours:
            if cv2.contourArea(c) > area_threshold:
                found = True
                rect_par = cv2.boundingRect(c)
                x, y, w, h = add_padding(rect_par, padding_ratio, res)
                parent.cur_imgs.append(frame[y:y+h, x:x+w]) # .copy()
                parent.cur_bboxes.append((x, y, w, h))
                
        if found:
            print("found_obj")
            parent.found_obj()
            
    # print(parent.state)
    parent.timeout()

def out_detect_fn(parent):
    pass


