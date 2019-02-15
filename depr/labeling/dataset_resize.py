import cv2
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("inp", help="Folder name to resize images in")
parser.add_argument("out", help="Folder name to save the cropped images")
args = parser.parse_args()

dir = args.inp + "/"
out_dir = args.out

if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

images = []

for file in os.listdir(dir):
    if file.endswith(".jpg") or file.endswith(".jpeg"):
        images.append(dir + file)

total = len(images)
        
count = 0
for im in images:

    #print(im)
    image = cv2.imread(im, -1)
    height, width = image.shape[:2]
    #print(image.shape[:2])

    height_target, width_target = (1080, 1920)
    scaling = max(width_target / width, height_target / height)

    if scaling > 1:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC
    
    res = cv2.resize(image, (0, 0), fx=scaling, fy=scaling, interpolation=interpolation)
    new_height, new_width = res.shape[:2]

    base = im.split("/")[1].split(".")[0]
    filename = out_dir + "/" + base + ".jpg"
    cv2.imwrite(filename, res[0:height_target, 0:width_target])  

    if new_height > height_target:
        filename = out_dir + "/" + base + "-alt.jpg"
        cv2.imwrite(filename, res[new_height-height_target:new_height, 0:width_target])  

    elif new_width > width_target:
        filename = out_dir + "/" + base + "-alt.jpg"
        cv2.imwrite(filename, res[0:height_target, new_width-width_target:new_width])  

    #exit()
    #cv2.imshow("image", res[0:height_target, 0:width_target])
    #cv2.moveWindow("image", 100, 20)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    count += 1
    print("Processed ", count, "/", total) 
