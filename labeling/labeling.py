# import the necessary packages
import argparse
import shutil
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

refPt = []
found_drone = False
cropping = False
cv_im = None
clone = None
previous_image = None
tracking = False
tracker = cv2.TrackerCSRT_create()

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, cv_im, clone, tracker, tracking
    drawing = False

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        tracker.clear()
        tracker = cv2.TrackerCSRT_create()
        tracking = False
        refPt = [(format_x(x), format_y(y))]
        cropping = True

    elif cropping and event == cv2.EVENT_MOUSEMOVE:
        if refPt:
            cv_im = clone.copy()
            orig_x = refPt[0][0]
            orig_y = refPt[0][1]
            cur_x = format_x(x)
            cur_y = format_y(y)
            cv2.rectangle(cv_im, (orig_x, orig_y), (cur_x, cur_y), (0,255,0), 2)
            cv2.imshow("image", cv_im)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((format_x(x), format_y(y)))
        #print(x, y)
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(cv_im, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", cv_im)
        
def format_x(x):
    return min(x, 1920) if (x > 0) else 0
    
def format_y(y):
    return min(y, 1080) if (y > 0) else 0

def auto_bbox():
    '''
    Detects the potential drones in the image by simple Canny edge detection.
    Input: rgb image
    Output: Bounding box top left and bottom right coordinates ((x1, y1), (x2, y2))
    '''
    global tracker, tracking, previous_image
    
    if (previous_image is None) or (not found_drone):
        return [(0, 0), (0, 0)]

    # previous user entry found a drone
    if (not tracking):
        success_init = tracker.init(previous_image, format_xywh(refPt))
        if not success_init:
            return [(0, 0), (0, 0)]    

    tracking, drone_bbox = tracker.update(clone)
    if tracking:
        x, y, w, h = drone_bbox
        p1 = (int(x), int(y))
        p2 = (int(x + w), int(y + h))
        return [p1, p2]
    else:
        tracker.clear()
        tracker = cv2.TrackerCSRT_create()
        return [(0, 0), (0, 0)] 
        

def print_help():
    print("Usage:")
    print("If the bounding box prediction is right, press 's' to save it")
    print("If the estimate is wrong, draw the correct bounding box with the mouse.")
    print("If the image does not belong to a drone, press 'd'")
    print("Program terminates after cycling through all images in the 'images' folder")


def add_label(outfile, im_name, prob, refPt):
    '''
    Append the entry to the specified label file
    '''
    x, y, w, h = format_xywh(refPt)
    
    entry_list = [im_name, prob, str(y), str(x), str(w), str(h)]
    f = open(outfile, 'a')
    writer = csv.writer(f, delimiter=',', quotechar='"')
    writer.writerow(entry_list)
    return

def format_xywh(refPt):
    top_left_x  = format_x(min(refPt[0][0], refPt[1][0]))
    top_left_y = format_y(min(refPt[0][1], refPt[1][1]))
    bot_right_x = format_x(max(refPt[0][0], refPt[1][0]))
    bot_right_y = format_y(max(refPt[0][1], refPt[1][1]))
    
    w = abs(bot_right_x - top_left_x)  # absolute value not really necessary
    h = abs(bot_right_y - top_left_y)  # absolute value not really necessary
    
    return top_left_x, top_left_y, w, h

def unprocessed_images(outfile):
    # Set file names to read from / save into
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
            if entry_list:
                image_path = entry_list[0]
                existing_entries.add(image_path)

    # Populate the list of images to make a list for
    images = []
    total = 0
    for file in os.listdir(args.folder):
        if (file.endswith(".jpg") or file.endswith(".jpeg")) and (args.folder + "/" + file not in existing_entries):
            images.append(args.folder + "/" + file)
        total += 1
    images.sort(key=extract_image_number)
    return images, total
    
def extract_image_number(file_name):
    return int(file_name.split('(')[1].split(')')[0])
    

def main():
    global refPt, cropping, cv_im, clone, previous_image, found_drone

    if args.output is not None:
        outfile = args.output
    else:
        outfile = args.folder + ".csv"

    lst_unlabeled_images, total_num_images = unprocessed_images(outfile)
    total_num_processed = total_num_images - len(lst_unlabeled_images)
    # Check if the entire dataset have been processed before
    if not lst_unlabeled_images:
        print("All images in the given folder have been labeled. Exiting.. ")
        exit()
    else:
        print_help()

    for image in lst_unlabeled_images:
        if clone is not None:
            previous_image = clone.copy()
        cv_im = cv2.imread(image)
        clone = cv_im.copy()
        refPt = auto_bbox()
        cv2.rectangle(cv_im, refPt[0], refPt[1], (0, 255, 0), 2)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.moveWindow("image", 1366, 0)
        cv2.setMouseCallback("image", click_and_crop)
        cv2.putText(img=cv_im,
                    text=str(total_num_processed) + ' / ' + str(total_num_images),
                    org=(1200, 900),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=3,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA)
                    
        # keep looping until the 'q' key is pressed
        cycle = True
        while cycle:
            # display the image and wait for a keypress
            cv2.imshow("image", cv_im)
            
            key = cv2.waitKey(0) & 0xFF
            
            # if the 'x' key is pressed, the image is not viable. Move image to deleted folder
            if key == ord("x"):
                image_folder = image.split('/')[0]
                del_dir = "deleted/" + image_folder
                if not os.path.isdir(del_dir):
                    if not os.path.isdir("deleted"):
                        os.mkdir("deleted")
                    os.mkdir(del_dir)
                shutil.move(image, del_dir)
                found_drone = False
                break

            # if the 'd' key is pressed, there is no drone in the image
            elif key == ord("d"):
                zeros = [(0, 0), (0, 0)]
                add_label(outfile, image, "0", zeros)
                found_drone = False
                break

            # if the 's' key is pressed, save the bbox coordinates and break from the loop
            elif key == ord("s"): 
                add_label(outfile, image, "1", refPt)
                found_drone = True
                break

            # if the 'q' key is pressed, stop labeling and exit
            elif key == ord("q"):
                cycle = False
        
        total_num_processed += 1
        if not cycle:
            break

    # close all open windows
    cv2.destroyAllWindows()

    print("Folder completed. Exiting..")


if __name__ == '__main__':
    main()
