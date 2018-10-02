# import the necessary packages
import argparse
import cv2
import csv
import os
import re
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
# as well as path for images and list of image names

parser = argparse.ArgumentParser()
parser.add_argument("folder", help="Folder name with images to make labels for")
parser.add_argument("-o", "--output", default=None, help="Image label csv file name. Default is <folder>.csv")
args = parser.parse_args()

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, cv_im
    drawing = False

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif cropping and event == cv2.EVENT_MOUSEMOVE:
        if refPt:
            cv_im = clone.copy()
            orig_x = refPt[0][0]
            orig_y = refPt[0][1]
            cv2.rectangle(cv_im, (orig_x, orig_y), (x, y), (0,255,0), 2)
            cv2.imshow("image", cv_im)

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
    print("If the estimate is wrong, draw the correct bounding box with the mouse.")
    print("If the image does not belong to a drone, press d")
    print("Program terminates after cycling through all images in the 'images' folder")


def add_label(outfile, im_name, prob, top_x, top_y, bot_x, bot_y):
    '''
    Append the entry to the specified label file
    '''
    w = bot_x - top_x
    h = bot_y - top_y
    entry_list = [im_name, prob, str(top_y), str(top_x), str(w), str(h)]
    f = open(outfile, 'a')
    writer = csv.writer(f, delimiter=',', quotechar='"')
    writer.writerow(entry_list)
    return

images = []
refPt = []
cropping = False

# Set file names to read from / save into
dir = args.folder
if args.output is not None:
    outfile = args.output
else:
    outfile = args.folder + ".csv"

# Check if any entries have been added before
existing_entries = set()
if os.path.isfile(outfile):
    file = open(outfile, 'r')
    reader = csv.reader(file, delimiter=',', quotechar='"')
    for entry_list in reader:
        image_path = entry_list[0]
        existing_entries.add(image_path)

# print(existing_entries)

# Populate the list of images to make a list for
for file in os.listdir(dir):
    if (file.endswith(".jpg") or file.endswith(".jpeg")) and (dir + "/" + file not in existing_entries):
        images.append(dir + "/" + file)

# Check if the entire dataset have been processed before
if not images:
    print("All images in the given folder have been labeled. Exiting.. ")
    exit()

print_help()

for image in images:
    cv_im = cv2.imread(image)
    clone = cv_im.copy()
    refPt = auto_bbox(cv_im)
    cv2.rectangle(cv_im, refPt[0], refPt[1], (0, 255, 0), 2)

    cv2.namedWindow("image")
    cv2.moveWindow("image", 20, 20)
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    cycle = True
    while cycle:
        # display the image and wait for a keypress
        cv2.imshow("image", cv_im)
        key = cv2.waitKey(0) & 0xFF

        # if the 'x' key is pressed, the image is not viable. Delete the image
        if key == ord("x"):
            os.remove(image)
            break

        # if the 'd' key is pressed, there is no drone in the image
        elif key == ord("d"):
            add_label(outfile, image, "0", 0, 0, 0, 0)
            break

        # if the 's' key is pressed, save the bbox coordinates and break from the loop
        elif key == ord("s"):
            add_label(outfile, image, "1", refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1])
            break

        # if the 'q' key is pressed, stop labeling and exit
        elif key == ord("q"):
            cycle = False

    if not cycle:
        break

# close all open windows
cv2.destroyAllWindows()

print("Folder completed. Exiting..")
