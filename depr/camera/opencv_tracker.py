from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("output", help="Output video file name")
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args.filename, fourcc, 20.0, (640, 480))
    
while True:
    # grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    # check to see if we have reached the end of the stream
    if frame is None:
        break

    print(frame.shape)
    
    (success, boxes) = trackers.update(frame)
    print(success)
    
    # loop over the bounding boxes and draw then on the frame
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    out.write(frame)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
                
        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
    if key == ord("s"):
	# select the bounding box of the object we want to track (make
	# sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            
        # create a new object tracker for the bounding box and add it
	# to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)

    elif key == ord("q"):
        break
            
# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
    
# otherwise, release the file pointer
else:
    vs.release()
    
# close all windows
out.release()
cv2.destroyAllWindows()
