# import the necessary packages
import cv2
import argparse
from tempfile import NamedTemporaryFile
import shutil
import csv

parser = argparse.ArgumentParser()
parser.add_argument("annotations", help="File name with annotations")
args = parser.parse_args()

output_file = open("incorrect_labels.csv", 'x')

with open(args.annotations, 'r') as annotations:

    for entry in annotations.readlines():
        entry_list = entry.split(',')
        image_path = entry_list[0]
        drone_present = entry_list[1]
        x = int(entry_list[2])
        y = int(entry_list[3])
        w = int(entry_list[4])
        h = int(entry_list[5])

        cv_im = cv2.imread(image_path)
        cv2.rectangle(cv_im, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.namedWindow("image")
        cycle = True
        while cycle:
            # display the image and wait for a keypress
            cv2.imshow("image", cv_im)
            key = cv2.waitKey(0)
            print(key)

            # if the 'x' key is pressed, the image is not viable. Delete the image
            if key == ord("x"):
                correct_label ^= drone_present
                csv_entry = ','.join([image_path, drone_present, correct_label])
                output_file.write(csv_entry) + "\n"
                break

            # if the 'd' key is pressed, there is no drone in the image
            elif key == ord("s"):
                break

            # if the 'q' key is pressed, stop labeling and exit
            elif key == ord("q"):
                cycle = False

        if not cycle:
            break



# close all open windows
cv2.destroyAllWindows()
