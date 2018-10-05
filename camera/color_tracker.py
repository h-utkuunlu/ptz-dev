import cv2
import imutils
from time import sleep, time
from camera_controls import PTZOptics20x
from async_reader import Camera

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

class PIDController:
    def __init__(self, kp, kd, ki, T, omega_c):
        self.past = {
            "err": 0.0,
            "diff": 0.0,
            "filt": 0.0,
            "integ": 0.0
        }
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.T = T
        self.omega_c = omega_c

        self.minval = -24
        self.maxval = 24
        
    # Controls equations - bilinear approximation
    @staticmethod
    def _df_bil(diff_1, err, err_1, T):
        return -diff_1 + (err - err_1) * (2 / T)
    @staticmethod
    def _filt_bil(filt_1, diff, diff_1, T, omega_c):
        return ((2 - T*omega_c)/(2 + T*omega_c))*filt_1 + T*omega_c*(diff + diff_1)/(2+T*omega_c)
    @staticmethod
    def _integ_bil(integ_1, err, err_1, T):
        return integ_1 + (err + err_1)*(T/2)

    def compute(self, err):
        diff = self._df_bil(self.past["diff"], err, self.past["err"], self.T)
        filt = self._filt_bil(self.past["filt"], diff, self.past["diff"], self.T, self.omega_c)
        integ = self._integ_bil(self.past["integ"], err, self.past["err"], self.T)

        pid_out = self.kp*err + self.kd*filt + self.ki*integ

        self.past["err"] = err
        self.past["diff"] = diff
        self.past["filt"] = filt

        if (pid_out > self.minval and pid_out < self.maxval):
            self.past["integ"] = integ

        return pid_out

def control(errors, pan_pid, tilt_pid, ptz):

    dur = 0.001
    
    pan_command = pan_pid.compute(errors[0]) # positive means turn left
    tilt_command = tilt_pid.compute(errors[1]) # positive means move up
    
    pan_speed = limit(pan_command, 24)
    tilt_speed = limit(tilt_command, 18)
    
    if pan_speed == 0 and tilt_speed == 0:
        ptz.stop()
        sleep(dur)

    elif pan_speed == 0:
        if tilt_command == abs(tilt_command):
            ptz.up(tilt_speed)
            sleep(dur)
        else:
            ptz.down(tilt_speed)
            sleep(dur)
    elif tilt_speed == 0:
        if pan_command == abs(pan_command):
            ptz.left(pan_speed)
            sleep(dur)
        else:
            ptz.right(pan_speed)
            sleep(dur)

    elif abs(pan_command) == pan_command and abs(tilt_command) == tilt_command:
        ptz.left_up(pan_speed, tilt_speed)
        sleep(dur)

    elif abs(pan_command) == pan_command and abs(tilt_command) != tilt_command:
        ptz.left_down(pan_speed, tilt_speed)
        sleep(dur)

    elif abs(pan_command) != pan_command and abs(tilt_command) == tilt_command:
        ptz.right_up(pan_speed, tilt_speed)
        sleep(dur)

    elif abs(pan_command) != pan_command and abs(tilt_command) != tilt_command:
        ptz.right_down(pan_speed, tilt_speed)
        sleep(dur)

    return pan_speed, tilt_speed

def limit(val, max):
    retval = max if abs(val) > max else abs(int(val))
    return retval

def errors(center, width, height):
    return ((width//2 - center[0])/50, (height//2 - center[1])/30) # Pos. pan error: right. Pos. tilt error: down

host = "192.168.1.40"
port = 5678

ptz = PTZOptics20x(host, port)
ptz.init()

width = 1280
height = 720
source = Camera(2, width, height)

cv2.namedWindow("cam")
cv2.moveWindow("cam", 20, 20)

pan_pid = PIDController(1.4, 0.1, 0, 1/50, 2*3.14*10)
tilt_pid = PIDController(1.4, 0.1, 0, 1/50, 2*3.14*10)

start_time = time()

counter = 0
count_limit = 200

#while counter < count_limit:
while True:
    frame = None
    frame = source.cvreader.Read()

    if frame is None:
        continue
    
    out = find_object(frame)

    if out is None:
        cv2.imshow("cam", frame)
        continue
    
    #print(time() - start_time)
    #start_time = time()
    
    center, x, y, radius = out

    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)

    print(control(errors(center, width, height), pan_pid, tilt_pid, ptz))
    counter += 1
    
    cv2.imshow("cam", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

source.cvreader.Stop()
source.cvcamera.release()
cv2.destroyAllWindows()

ptz.stop()
sleep(1)
