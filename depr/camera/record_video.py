import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="File name for the output video")
parser.add_argument("-u", "--usb", type=int, default=0, help="USB device no for the camera to record the video")
args = parser.parse_args()

cap = cv2.VideoCapture(args.usb)
cap.set(3, 1920)
cap.set(4, 1080)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.filename, fourcc, 30.0, (1920, 1080))
cv2.namedWindow("out")
cv2.moveWindow("out", 20, 20)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.imshow("out", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

