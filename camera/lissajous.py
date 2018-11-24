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

def get_lissajous(t, freq, radius):
    x_factor = cos(2*pi*freq * t)
    y_factor = sin(2*pi*freq * t)

    pan_pos = int(radius * x_factor)
    if pan_pos < 0:
        pan_pos = (2**16 - 1) + pan_pos

    tilt_pos = int(radius * y_factor)
    if tilt_pos < 0:
        tilt_pos = (2**16 - 1) + tilt_pos

    return (pan_pos, tilt_pos)

freq = float(args.freq)
radius = 500
gui_mode = args.gui
width = 1280
height = 720
# pan_tilt_threshold = 3000  # not exact, simply provides a threshold to test RHP or LHP

ptz = PTZOptics20x(args.host, args.port)
ptz.init()
source = Camera(args.dev, width, height)
pan_pid = PIDController(1.4, 0.1, 0.0, 1.0/50.0, 2*3.14*10)
tilt_pid = PIDController(1.4, 0.1, 0.0, 1.0/50.0, 2*3.14*10)

if gui_mode:
    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 20, 20)

# Start Position
print("Initialize start position")
ptz.home()
sleep(1.0)
ptz.goto(500, 0, 24)
sleep(1.0)

# Move camera in Circle
start_time = time()
while True:

    elapsed_time = time() - start_time
    pan_pos, tilt_pos = get_lissajous(elapsed_time, freq, radius)  # Determine target abs. pos.
    

    # print(pan_pos, tilt_pos)
    ptz.goto(pan_pos, tilt_pos, 24)
    sleep(0.1)


source.cvreader.Stop()
source.cvcamera.release()

if gui_mode:
    cv2.destroyAllWindows()

ptz.stop()
sleep(1)
print("Program complete. Exiting..")
