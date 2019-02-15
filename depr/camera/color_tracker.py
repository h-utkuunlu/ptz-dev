import cv2
import imutils
from time import sleep, time
from camera_controls import PTZOptics20x
from async_reader import Camera
from controller import PIDController, control, limit, errors_pt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-H', '--host', default='192.168.1.40', help="Host address for the camera for PTZ control")
parser.add_argument('-p', '--port', default=5678, help="Port for TCP comms PTZ control", type=int)
parser.add_argument('-d', '--dev', default=0, help='Camera USB device location for OpenCV', type=int)
parser.add_argument('-c', '--count', default=600, help='Number of frames to run for non-GUI mode', type=int)
parser.add_argument('-g', '--gui', help='GUI mode', action='store_true')
args = parser.parse_args()

def find_object(frame):

    lower = (100, 50, 50)
    upper = (160, 255, 255)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            return [center, x, y, radius]

def control_zoom(error, zoom_pid, ptz):

    dur = 0.001

    zoom_command = zoom_pid.compute(error) # positive means zoom in
    zoom_speed = limit(zoom_command, 1)

    if not zoom_speed:
        ptz.zoomstop()
        sleep(dur)

    if zoom_command > 0:
        ptz.zoomin(zoom_speed)
        sleep(dur)
    elif zoom_command < 0:
        ptz.zoomout(zoom_speed)
        sleep(dur)

    return zoom_speed

def error_zoom(size, height):
    target_size = float(height)/3.0  # no specific reason for 3
    return (target_size - size)/30  # no specific reason for 30

gui_mode = args.gui

ptz = PTZOptics20x(args.host, args.port).init()

width = 1280
height = 720
source = Camera(args.dev, width, height)

if gui_mode:
    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 20, 20)

pan_pid = PIDController(1.2, 0.1, 0.1, 1/50, 2*3.14*10)
tilt_pid = PIDController(1.2, 0.1, 0.1, 1/50, 2*3.14*10)
zoom_pid = PIDController(1.2, 0.1, 0.1, 1/50, 2*3.14*10)

start_time = time()

counter = 0
count_limit = args.count

while True:
    frame = None
    frame = source.cvreader.Read()

    if frame is None:
        continue

    out = find_object(frame)

    if out is None:
        if gui_mode:
            cv2.imshow("cam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        continue

    center, x, y, radius = out

    if gui_mode:
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        cv2.imshow("cam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    pan_tilt_error = errors_pt(center, width, height)
    zoom_error = error_zoom(2*radius, height)
    control(pan_tilt_error, pan_pid, tilt_pid, ptz)
    control_zoom(zoom_error, zoom_pid, ptz)

    print("Frame Rate:", int(1/ (time() - start_time)))

    start_time = time()
    counter += 1

    if not gui_mode and counter > count_limit:
        break

source.cvreader.Stop()
source.cvcamera.release()

if gui_mode:
    cv2.destroyAllWindows()

ptz.stop()
sleep(1)
print("Program complete. Exiting..")
