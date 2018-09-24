import cv2
import os
import numpy as np
import argparse

# Add part to read dir list to load all images after figuring out
parser = argparse.ArgumentParser()
parser.add_argument('image', help="Image filename to be loaded")
args = parser.parse_args()

#filename = "images/1.jpg"

img = cv2.imread(args.image, 0)
#orig = cv2.imread(args.image, -1)
edges = cv2.Canny(img, 100, 200)
kernel = np.ones((5, 5), np.uint8)

for i in range(3):
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
edges = cv2.dilate(edges, kernel, iterations=1)

im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
if len(contours) != 0:
    print("I was run!")
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("result", np.hstack([img, edges]))
cv2.moveWindow("result", 50, 50)
#cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
