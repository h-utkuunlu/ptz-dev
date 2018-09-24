# import the necessary packages
import argparse
import cv2
import os
import re
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping
    
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        #print(x, y)
        cropping = True
            
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
	# record the ending (x, y) coordinates and indicate that
	# the cropping operation is finished
        refPt.append((x, y))
        #print(x, y)
        cropping = False
            
	# draw a rectangle around the region of interest
        cv2.rectangle(cv_im, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", cv_im)

def auto_bbox(image):
    '''
    Detects the potential drones in the image by simple Canny edge detection.
    Input: rgb image
    Output: Bounding box top left and bottom right coordinates ((x1, y1), (x2, y2))
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5, 5), np.uint8)

    for i in range(3):
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
    edges = cv2.dilate(edges, kernel, iterations=1)

    _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return [(x, y), (x+w, y+h)]

    else:
        return [(0, 0), (0, 0)]

def print_help():
    print("Usage:")
    print("If the bounding box prediction is right, press s to save it")
    print("If the estimate is wrong, press r to reset the screen, and draw the bounding box with the mouse. Box shows only after you release the mouse button")
    print("If the image does not belong to a drone, press x")
    print("Program terminates after cycling through all images in the 'images' folder")
    
images = []
dir = "images/"

for file in os.listdir(dir):
    if file.endswith(".jpg") or file.endswith(".jpeg"):
        images.append(dir + file)
#print(images)

# load the image, clone it, and setup the mouse callback function
print_help()

bboxes = []
for image in images:
    cv_im = cv2.imread(image)
    clone = cv_im.copy()
    refPt = auto_bbox(cv_im)
    cv2.rectangle(cv_im, refPt[0], refPt[1], (0, 255, 0), 2)
        
    cv2.namedWindow("image")
    cv2.moveWindow("image", 2000, 20)
    cv2.setMouseCallback("image", click_and_crop)
    
    # keep looping until the 'q' key is pressed
    while True:
	# display the image and wait for a keypress
        cv2.imshow("image", cv_im)
        key = cv2.waitKey(0) & 0xFF
            
	# if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            cv_im = clone.copy()

        # if the 'x' key is pressed, there is no drone in the image
        if key == ord("x"):
            bboxes.append([(0, 0), (0, 0)])
            break

        # if the 's' key is pressed, save the bbox coordinates and break from the loop
        elif key == ord("s"):
            bboxes.append(refPt)
            break
                   
# close all open windows
cv2.destroyAllWindows()
print(bboxes)
