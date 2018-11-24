import cv2
import imutils
from time import sleep, time
from camera_controls import PTZOptics20x
from async_reader import Camera
from controller import PIDController, control, limit
from numpy import pi, cos, sin
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-H', '--host', default='192.168.1.40', help="Host address for the camera for PTZ control")
parser.add_argument('-p', '--port', default=5678, help="Port for TCP comms PTZ control", type=int)
parser.add_argument('-d', '--dev', default=0, help='Camera USB device location for OpenCV', type=int)
parser.add_argument('-f', '--freq', default=0.05, help='Frequency of lissajous loop', type=float)
parser.add_argument('-g', '--gui', help='GUI mode', action='store_true')
args = parser.parse_args()

def lissajous(t, freq, radius):
    x_factor = cos(2*pi*freq * t)
    y_factor = sin(2*pi*freq * t)

    pan_pos = int(radius * x_factor)
    tilt_pos = int(radius * y_factor)

    return (pan_pos, tilt_pos)

def normalize(position):
    if position[0] == -1:
        return (0,0), False
    pan_pos, tilt_pos = position
    if pan_pos > pan_tilt_threshold:
        pan_pos = pan_pos - (2**16 - 1)
    if tilt_pos > pan_tilt_threshold:
        tilt_pos = tilt_pos - (2**16 - 1)
    return (pan_pos, tilt_pos), True

def errors(position, position_target):
    pan_pos, tilt_pos = position
    pan_target, tilt_target = position_target

    pan_diff = pan_pos - pan_target
    tilt_diff = tilt_pos - tilt_target
    # print("Pan Diff: ", pan_diff)
    # print("Tilt Diff: ", tilt_diff)

    return (pan_diff/50, tilt_diff/30) # Pos. pan error: right. Pos. tilt error: down


freq = float(args.freq)
radius = 500
pan_tilt_threshold = 3000  # not exact, simply provides a threshold to test RHP or LHP
gui_mode = args.gui
width = 1280
height = 720

ptz = PTZOptics20x(args.host, args.port)
ptz.init()
source = Camera(args.dev, width, height)


pan_pid = PIDController(1.2, 0.3, 0.01, 1.0/50.0, 2*3.14*10)
tilt_pid = PIDController(1.2, 0.3, 0.01, 1.0/50.0, 2*3.14*10)

if gui_mode:
    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 20, 20)

print("Go Home")
ptz.home()
sleep(1)
ptz.goto(500, 0, 24)
sleep(1)

print("Done")

start_time = time()
while True:
    # frame = None
    # frame = source.cvreader.Read()
    #
    # if frame is None:
    #     continue
    #
    # if gui_mode:
    #     cv2.imshow("cam", frame)
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break

    # while True:
    #     sleep(0.05)
    #     elapsed = time() - start_time
    #     position_targ = lissajous(elapsed, freq, radius)
    #     print("Targ: ", position_targ)

    position, val_position = normalize(ptz.get_pan_tilt_position())
    sleep(0.5)
    if (val_position):
        # elapsed = time() - start_time
        # position_targ = lissajous(elapsed, freq, radius)
        # error = errors(position, position_targ)
        print("Curr: ", position)
        ptz.left(8)
        # print(ptz.read())
        # print(ptz.read())
        # print("Targ: ", position_targ)
        # print("Error: ", error)

        # speeds = control(error, pan_pid, tilt_pid, ptz)
        # print("Speeds: ", speeds)

        # ptz.read()


source.cvreader.Stop()
source.cvcamera.release()

if gui_mode:
    cv2.destroyAllWindows()

ptz.stop()
sleep(1)
print("Program complete. Exiting..")
