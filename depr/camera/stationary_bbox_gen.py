import cv2
import numpy as np
import argparse
import time

def add_padding(rect_par, ratio, dims=(1280, 720)):
    x, y, w, h = rect_par
    padx = int(w*ratio)
    pady = int(h*ratio)

    if x-padx < 0 or y-pady < 0 or x+w+2*padx > dims[0] or y+h+2*pady > dims[0]:
        return (x, y, w, h)
    else:
        return (x-padx, y-pady, w+2*padx, h+2*pady)

def im_dim(image):
    return tuple(reversed(image.shape[:2]))
    
def min_resize(source_image, dim=224):
    img = source_image.copy()
    min_dim = min(im_dim(img))
    factor = dim / min_dim

    if factor > 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    
    img = cv2.resize(img, (0, 0), fx=factor, fy=factor, interpolation=interpolation)
    return img
    
parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to input video")
args = parser.parse_args()

vid = cv2.VideoCapture(args.input)

fgbg = cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=20)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel = np.ones((5, 5), np.uint8)

cv2.namedWindow('frame')
# cv2.namedWindow('fgmask')

res = (1280, 720)
init_count = 0

clear_count = 0
while vid.isOpened():
    ret, frame = vid.read()

    if ret == False:
        break

    frame = cv2.resize(frame, res, interpolation=cv2.INTER_AREA)
    res = im_dim(frame)

    if clear_count < 5:
        clear_count += 1
        continue
    
    #print(res)
    #exit()
    if init_count < 20:
        fgmask = fgbg.apply(frame)
        init_count += 1
        continue
    
    start = time.time()
    # Operations on the frame
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=10)
    fgmask = cv2.erode(fgmask, kernel, iterations=8)
    #fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    _, contours, _ = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    #_, contours, _ = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # No difference btw algorithms

    #boundingRect = []
    areas = []

    count = 0
    for c in contours:
        #areas.append(cv2.contourArea(c))
        if cv2.contourArea(c) > 100:
            #boundingRect.append(cv2.boundingRect(c))
            rect_par = cv2.boundingRect(c)
            x, y, w, h = add_padding(rect_par, 0.2, res)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            areas.append(w*h)
            
            ### Extract and show
            #image = frame[y:y+h, x:x+w]
            #image = min_resize(image)
            
            #name = "extract-%d" % count
            #cv2.imshow(name, image)
            #count += 1
            
    #frame = cv2.bitwise_and(frame, frame , mask=fgmask)
    
    dur = time.time() - start
    print("Areas:", areas)
    
    
    # Display frame
    #fgmask = cv2.resize(fgmask, res, interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', frame)
    #cv2.imshow('fgmask', fgmask)

    # UI
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
