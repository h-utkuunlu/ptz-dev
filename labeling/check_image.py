# import the necessary packages
import argparse
import cv2
import csv

parser = argparse.ArgumentParser()
parser.add_argument("entry", help="Image entry number")
parser.add_argument("annotations", help="Annotations File")
args = parser.parse_args()

def main():
    annotations = open(args.annotations, 'r')
    reader = csv.reader(annotations, delimiter=',', quotechar='"')
    for _ in range(int(args.entry)-1):
        next(reader)

    entry_list = next(reader)
    image_path = entry_list[0]
    drone_present = entry_list[1]
    row = int(entry_list[2])
    col = int(entry_list[3])
    w = int(entry_list[4])
    h = int(entry_list[5])

    cv_im = cv2.imread(image_path)
    cv2.rectangle(cv_im, (col, row), (col+w, row+h), (0, 255, 0), 3)
    cv2.namedWindow("image")
    cv2.moveWindow("image", 1366, 0)

    while True:
        # Display the image and wait for a keypress
        cv2.imshow("image", cv_im)
        key = cv2.waitKey(0)

        # If the 'q' key is pressed, stop labeling and exit
        if key == ord("q"):
            break

    annotations.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
