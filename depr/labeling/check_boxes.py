# import the necessary packages
import argparse
import cv2
import csv
import os
from tempfile import NamedTemporaryFile
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("annotations", help="File name with annotations")
args = parser.parse_args()

refPt = []
cropping = False
cv_im = None
clone = None


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, cv_im, clone
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
            cv2.rectangle(cv_im, (orig_x, orig_y), (x, y), (0,255,0), 1)
            cv2.imshow("image", cv_im)

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        #print(x, y)
        cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(cv_im, refPt[0], refPt[1], (0, 255, 0), 1)
        cv2.imshow("image", cv_im)

def check_annotations():
    global refPt, cropping, cv_im, clone
    # Output file that stores the fixes
    image_folder_name = args.annotations[:-4]  # folder of image assumed from annotations name
    annotations = open(args.annotations, 'r')
    reader = csv.reader(annotations, delimiter=',', quotechar='"')

    try:
        print("Creating out out file: ", image_folder_name + "_verified.csv")
        output_file = open(image_folder_name + "_verified.csv", 'x')  # create
    except FileExistsError:
        output_file = open(image_folder_name + "_verified.csv", 'r')  # append
        last_image_path = list(csv.reader(output_file, delimiter=',', quotechar='"'))[-1][0]
        output_file.close()
        for entry_list in reader:
            if entry_list[0] == last_image_path:
                break
        output_file = open(image_folder_name + "_verified.csv", 'a')  # append

    writer = csv.writer(output_file, delimiter=',', quotechar='"')

    for entry_list in reader:
        # Extract annotation details
        image_path = entry_list[0]
        drone_present = entry_list[1]
        row = int(entry_list[2])
        col = int(entry_list[3])
        w = int(entry_list[4])
        h = int(entry_list[5])

        # Open image and current bounding box with OpenCV
        cv_im = cv2.imread(image_path)
        clone = cv_im.copy()
        cv2.rectangle(cv_im, (col, row), (col+w, row+h), (0, 255, 0), 3)
        cv2.namedWindow("image")
        cv2.moveWindow("image", 1366, 0)
        cv2.setMouseCallback("image", click_and_crop)

        # Cycle through the images until you exit
        cycle = True
        while cycle:
            # Display the image and wait for a keypress
            cv2.imshow("image", cv_im)
            key = cv2.waitKey(0)

            # If the 'x' key is pressed, the image is not viable. Delete the image
            if key == ord("x"):
                image_folder = image_path.split('/')[0]
                del_dir = "deleted/" + image_folder
                if not os.path.isdir(del_dir):
                    if not os.path.isdir("deleted"):
                        os.mkdir("deleted")
                    os.mkdir(del_dir)
                shutil.move(image_path, del_dir)
                break

            # if the 'd' key is pressed, then you edited the bounding box
            elif key == ord("e"):
                top_x, top_y = refPt[0][0], refPt[0][1]
                w = refPt[1][0] - top_x
                h = refPt[1][1] - top_y
                entry = [image_path, 1, top_y, top_x, w, h]
                writer.writerow(entry)
                break

            # if the 'd' key is pressed, there is no drone in the image
            elif key == ord("d"):
                entry_list = [image_path, 0, 0, 0, 0, 0]
                writer.writerow(entry_list)
                break

            # if the right-arrow is pressed, the entry is correct, so move on
            elif key == 83:
                writer.writerow(entry_list)
                break

            # If the 'q' key is pressed, stop labeling and exit
            elif key == ord("q"):
                cycle = False

        if not cycle:
            break

    # close all open windows
    cv2.destroyAllWindows()
    annotations.close()
    output_file.close()


if __name__ == '__main__':
    check_annotations()
