import cv2
import imutils
from time import sleep, time
from camera_controls import PTZOptics20x
from async_reader import Camera
from controller import PIDController, control, limit
from numpy import pi, cos, sin
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('-H', '--host', default='192.168.1.40', help="Host address for the camera for PTZ control")
parser.add_argument('-d', '--dev', default=0, help='Camera USB device location for OpenCV', type=int)
parser.add_argument('-f', '--freq', default=0.05, help='Frequency of lissajous loop', type=float)
parser.add_argument('-b', '--buffer', default=0.1, help='Delay time between commands', type=float)
parser.add_argument('-g', '--gui', help='GUI mode', action='store_true')
parser.add_argument('-p', '--plot', help='Plot results', action='store_true')
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

def get_zoom_target(t, freq):
    max_zoom = 15070  # max zoom position value
    offset = int(max_zoom/2)
    amp = max_zoom/4
    factor = sin(2*pi*freq * t)

    return int(amp * factor) + offset

def normalize(position):
    pan_pos, tilt_pos = position
    if pan_pos > pan_tilt_threshold:
        pan_pos = pan_pos - (2**16 - 1)
    if tilt_pos > pan_tilt_threshold:
        tilt_pos = tilt_pos - (2**16 - 1)
    return (pan_pos, tilt_pos)

# Initialize parameters
freq = float(args.freq)
radius = 800
gui_mode = args.gui
plot_mode = args.plot
width = 1280
height = 720
pan_tilt_threshold = 3000  # not exact, simply provides a threshold to test RHP or LHP
delay = args.buffer

# Camera and controller
ptz = PTZOptics20x(args.host)
ptz.init()
source = Camera(args.dev, width, height)
pan_pid = PIDController(1.4, 0.1, 0.0, 1.0/50.0, 2*3.14*10)
tilt_pid = PIDController(1.4, 0.1, 0.0, 1.0/50.0, 2*3.14*10)

# Open GUI
if gui_mode:
    cv2.namedWindow("cam")
    cv2.moveWindow("cam", 20, 20)

# Start Position
print("Initialize start position")
ptz.home()
sleep(2.0)
ptz.zoomto(int(15070/2))
# ptz.goto(radius, 0, 24)
sleep(3.0)

# Record camera values
if plot_mode:
    try:
        output_file = open("plot_freq_" + str(freq) + "_delay_" + str(args.buffer) + ".csv", 'x')  # create
    except FileExistsError:
        output_file = open("plot_freq_" + str(freq) + "_delay_" + str(args.buffer) + ".csv", 'w')
    writer = csv.writer(output_file, delimiter=',', quotechar='"')
    writer.writerow(["elapsed_time", "targ_zoom", "cur_zoom"])
    # writer.writerow(["elapsed_time", "pan_pos", "tilt_pos", "cur_pan_pos", "cur_tilt_pos"])

#
start_time = time()
while True:
    frame = source.cvreader.Read()
    if frame is None:
        continue
    elapsed_time = time() - start_time
    targ_zoom = get_zoom_target(elapsed_time, freq)
    cur_zoom, valid = ptz.get_zoom_position()
    targ_pan, targ_tilt = get_lissajous(elapsed_time, freq, radius)  # Determine target abs. pos.
    cur_pan, cur_tilt, valid = ptz.get_pan_tilt_position()
    if gui_mode:
        cv2.imshow("cam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if valid:
        n_targ_pan, n_targ_tilt = normalize((targ_pan, targ_tilt))
        n_cur_pan, n_cur_tilt = normalize((cur_pan, cur_tilt))
        if plot_mode:
            writer.writerow([elapsed_time, targ_zoom, cur_zoom])
            # writer.writerow([elapsed_time, n_targ_pan, n_targ_tilt, n_cur_pan, n_cur_tilt])
        print("Curr:", n_cur_pan, n_cur_tilt)
        # print("Targ:", n_pan, n_tilt)
        print("Curr:", cur_zoom)
        print("Targ:", targ_zoom)

    ptz.goto(targ_pan, targ_tilt, 24)
    sleep(delay)
    ptz.zoomto(targ_zoom)
    sleep(delay)



source.cvreader.Stop()
source.cvcamera.release()

if gui_mode:
    cv2.destroyAllWindows()

ptz.stop()
sleep(1)
print("Program complete. Exiting..")
