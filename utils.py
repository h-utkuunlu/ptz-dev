from threading import Thread
from threading import Lock
from time import sleep, time
import cv2
import re
import binascii
import socket
import fcntl
import os, sys
import errno
import numpy as np
import random
from transitions import Machine, State
from threading import Timer
from dnn import initialize_net, Resize

class SensibleWindows(object):
    def __init__(self, frame_name='main_window'):
        factor=1
        self.mw_x=0
        self.mw_y=0
        self.mw_w=int(1920*factor)
        self.mw_h=int(1080*(factor/2))
        self.ch3_fgmask=None
        self.async_frame=None
        self.frame=None
        self.frame_name = frame_name
        self.initialized = False
        cv2.namedWindow(self.frame_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.frame_name, self.mw_w, self.mw_h)
        cv2.moveWindow(self.frame_name,self.mw_x,self.mw_y)
        cv2.createButton('ABORT',abort,self)
        cv2.createButton('Reset',reset,self)
        self.ABORT = False
        self.RESET = False
        
    def init(self,frame):
        h,w,ch = frame.shape
        self.resizer = Resize(h)
        self.ch3_fgmask = np.zeros((h,w,ch),dtype=np.uint8)
        self.async_frame = np.zeros((h,h,ch),dtype=np.uint8)
        self.frame = frame
        self.initialized = True
        self.display()


    def update(self,frame=None,ch3_fgmask=None,async_frame=None):
        if frame is not None:
            self.frame = frame
        if ch3_fgmask is not None:
            self.ch3_fgmask = ch3_fgmask
        if async_frame is not None:
            self.async_frame=self.resizer(async_frame)
        self.display()
        
        
    def display(self):
        numpy_horiz_concat = np.concatenate((self.frame, self.ch3_fgmask, self.async_frame), axis=1)
        cv2.imshow(self.frame_name, numpy_horiz_concat)
        cv2.waitKey(1)
       
def abort(btn_state,parent):
    parent.ABORT=True
    
def reset(btn_state,parent):
    parent.RESET=True

        
class Flow(object):
    '''
    Finite state machine for ptz aerial tracking
    
 /''''''''''''''''''\
|       search       | <--------------.
 \__________________/                 |
    /|\          |                    |
     |timeout    |in_pos              |
     |          \|/                   |
 /''''''''''''''''''\                 |
|       detect       |                |
 \__________________/                 |
    /|\          |                    |
     |not_drone  |found_obj           |
     |          \|/                   |
 /''''''''''''''''''\                 |
|         id         |                |
 \__________________/                 |
     |                                |
     |drone                           |
    \|/                               |
 /''''''''''''''''''\  lost_track     |
|       track        |----------------'
 \__________________/
        

pip install transitions
'''
    states=[
        State('search'),
        State('detect'),
        State('id'),
        State('track')
    ]
    transitions = [
    { 'trigger': 'in_pos', 'source': 'search', 'dest': 'detect' },
    { 'trigger': 'timeout', 'source': 'detect', 'dest': 'search' },
    { 'trigger': 'found_obj', 'source': 'detect', 'dest': 'id' },
    { 'trigger': 'drone', 'source': 'id', 'dest': 'track' },
    { 'trigger': 'not_drone', 'source': 'id', 'dest': 'detect' },
    { 'trigger': 'lost_track', 'source': 'track', 'dest': 'search' }
    ]
    
    def __init__(self, model_path):

        # Variables
        self.timer_expir = True # bool for if timer expired
        self.obj_detected = False # bool for if object detected
        self.cur_imgs = []
        self.cur_bboxes = []
        self.drone_bbox = None
        self.bg_model = None
        self.timeout_interval = 5
        
        # Objects
        self.machine = Machine(self, states=Flow.states, transitions=Flow.transitions, initial='search', auto_transitions=False)
        self.camera = Camera(PIDController(),PIDController(),PIDController())
        self.tracker = cv2.TrackerCSRT_create()
        self.timer_obj = Timer(self.timeout_interval, self.expiry, ())
        self.network = initialize_net(model_path)
        self.gui = SensibleWindows()
        
        # Initialization routine
        init_count = 0
        while init_count < 5:
            _ = self.camera.cvreader.Read()
            init_count += 1

            
    def expiry(self):
        self.timer_expir = True    



class Camera:

    def __init__(self, pan_controller, tilt_controller, zoom_controller, usbdevnum=0, width=1280, height=720, fps=60, host='192.168.2.42', tcp_port=5678, udp_port=1259):

        # Camera params
        self.width = width
        self.height = height
        
        # Connect to PTZOptics camera for controls
        self.ptz = PTZOptics20x(host=host, tcp_port=5678, udp_port=1259).init()
        self.pan_controller = pan_controller
        self.tilt_controller = tilt_controller
        self.zoom_controller = zoom_controller
        
        # Open video stream as CV camera
        self.cvcamera = cv2.VideoCapture(usbdevnum)
        self.cvcamera.set(3, width)
        self.cvcamera.set(4, height)
        self.cvcamera.set(5, fps)
        
        # object running threads to get most recent frame and most recent zoom
        self.cvreader = CameraReaderAsync(self.cvcamera, self.ptz)

    def stop(self):
        self.cvreader.Stop()
        self.cvcamera.release()
        self.ptz.end()

    def control(self, pan_error, tilt_error):

        dur = 0.001

        pan_command = self.pan_controller.compute(pan_error) # positive means turn left
        tilt_command = self.tilt_controller.compute(tilt_error) # positive means move up

        pan_speed = self.limit(pan_command, 24)  # max speed for pan is 24
        tilt_speed = self.limit(tilt_command, 18)  # max speed for titlt is 18

        if pan_speed == 0 and tilt_speed == 0:
            self.ptz.stop()
            sleep(dur)

        elif pan_speed == 0:
            if tilt_command == abs(tilt_command):
                self.ptz.up(tilt_speed)
                sleep(dur)
            else:
                self.ptz.down(tilt_speed)
                sleep(dur)
        elif tilt_speed == 0:
            if pan_command == abs(pan_command):
                self.ptz.left(pan_speed)
                sleep(dur)
            else:
                self.ptz.right(pan_speed)
                sleep(dur)

        elif abs(pan_command) == pan_command and abs(tilt_command) == tilt_command:
            self.ptz.left_up(pan_speed, tilt_speed)
            sleep(dur)

        elif abs(pan_command) == pan_command and abs(tilt_command) != tilt_command:
            self.ptz.left_down(pan_speed, tilt_speed)
            sleep(dur)

        elif abs(pan_command) != pan_command and abs(tilt_command) == tilt_command:
            self.ptz.right_up(pan_speed, tilt_speed)
            sleep(dur)

        elif abs(pan_command) != pan_command and abs(tilt_command) != tilt_command:
            self.ptz.right_down(pan_speed, tilt_speed)
            sleep(dur)

        return pan_speed, tilt_speed

    @staticmethod
    def errors_pt(center, width, height):
        return (width//2 - center[0])/50, (height//2 - center[1])/30 # Pos. pan error: right. Pos. tilt error: down

    @staticmethod
    def limit(val, max):
        return max if abs(val) > max else abs(int(val))

    def control_zoom(self, error):

        dur = 0.001

        zoom_command = self.zoom_controller.compute(error) # positive means zoom in
        zoom_speed = self.limit(zoom_command, 1)

        if not zoom_speed:
            self.ptz.zoomstop()
            sleep(dur)

        if zoom_command > 0:
            self.ptz.zoomin(zoom_speed)
            sleep(dur)
        elif zoom_command < 0:
            self.ptz.zoomout(zoom_speed)
            sleep(dur)

        return zoom_speed

    @staticmethod
    def error_zoom(size, height):
        target_size = float(height)/5.0  # no specific reason for 8 factor
        return (target_size - size)/30  # no specific reason for 30


class CameraReaderAsync:

    class WeightedFramerateCounter:
        smoothing = 0.95
        startTime = 0
        framerate = 0

        def start(self):
            self.startTime = time()
            self.framerate = 0

        def tick(self):
            timeNow = time()
            if self.startTime == 0:
                self.startTime = timeNow
                return
            elapsed = 1.0 / (timeNow - self.startTime)
            self.startTime = timeNow
            self.framerate = (self.framerate * self.smoothing) + (elapsed * (1.0 - self.smoothing))

        def getFramerate(self):
            return self.framerate

    def __init__(self, videoSource, ptz):
        self.__lock = Lock()
        self.__telemetry_lock = Lock()
        self.__source = videoSource
        self.__ptz = ptz
        self.Start()

    def __ReadFrameAsync(self):
        while True:
            if self.__stopRequested:
                return
            validFrame, frame = self.__source.read()
            if validFrame:
                try:
                    self.__lock.acquire()
                    self.fps.tick()
                    self.__frame = frame
                    self.__lastFrameRead = False
                finally:
                    self.__lock.release()
                    
    def __ReadTelemetryAsync(self):
        while True:
            if self.__stopRequested:
                return

            validZoom, zoom = self.__ptz.get_zoom_position()
            sleep(0.005)
            validPT, pan, tilt = self.__ptz.get_pan_tilt_position()

            if validZoom and validPT:
                try:
                    #print("I'm putting telemetry in")
                    self.__telemetry_lock.acquire()
                    self.__zoom = (zoom/862.32)+1
                    self.__pan = pan
                    self.__tilt = tilt
                finally:
                    self.__telemetry_lock.release()
            sleep(0.01)
            
    def Start(self):
        self.__lastFrameRead = False
        self.__frame = None
        self.__stopRequested = False
        self.__validFrame = False
        self.__zoom = 0
        self.__pan = 0
        self.__tilt = 0
        self.fps = CameraReaderAsync.WeightedFramerateCounter()
        Thread(target=self.__ReadFrameAsync).start()
        Thread(target=self.__ReadTelemetryAsync).start()

    def Stop(self):
        self.__stopRequested = True

    # Return a frame if we have a new frame since this was last called.
    # If there is no frame or if the frame is not new, return None.
    def Read(self):
        try:
            self.__lock.acquire()
            if not self.__lastFrameRead:
                frame = self.__frame
                self.__lastFrameRead = True
                return frame

            return None
        finally:
            self.__lock.release()
            
    def ReadTelemetry(self):
        try:
            self.__telemetry_lock.acquire()
            return self.__pan, self.__tilt, self.__zoom
        finally:
            self.__telemetry_lock.release()
        
    # Return the last frame read even if it has been retrieved before.
    # Will return None if we never read a valid frame from the source.
    def ReadLastFrame(self):
        return self.__frame

class TCPCamera(object):


    def __init__(self, host, tcp_port=5678, udp_port=1259):
        """PTZOptics VISCA control class.

        :param host: TCP control host.
        :type host: str
        :param port: TCP control port.
        :type port: int
        """
        self._host = host
        self._tcp_port = tcp_port
        self._udp_port = udp_port

    def init(self):
        """Initializes camera object by establishing TCP control session.

        :return: Camera object.
        :rtype: TCPCamera
        """
        print("Connecting to camera...")
        # self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self._socket.setblocking(0)
        self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._tcp_socket.settimeout(0.6)
        self._udp_socket.settimeout(0.6)
        try:
            self._tcp_socket.connect((self._host, self._tcp_port))
        except:
            print("Could not connect to camera on tcp channel")
            return None
        try:
            self._udp_socket.connect((self._host, self._udp_port))
        except:
            print("Could not connect to camera on udp channel")
            return None
        print("Camera connected")
        self._tcp_socket.settimeout(0.2)
        self._udp_socket.settimeout(0.2)
        return self

    def command(self, com, channel):
        """Sends hexadecimal string to TCP control socket.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
        """
        if channel == "udp":
            try:
                self._udp_socket.send(binascii.unhexlify(com))
                return True
            except Exception as e:
                print(com, e)
                return False
        if channel == "tcp":
            try:
                self._tcp_socket.send(binascii.unhexlify(com))
                return True
            except Exception as e:
                print(com, e)
                return False

    def read(self, amount=1):
        total = ""
        while True:
            try:
                msg = binascii.hexlify(self._tcp_socket.recv(amount))
            except socket.timeout:
                print("No data from camera socket")
                break
            except socket.error:
                print("Camera socket read error.")
                break
            total = total + msg.decode("utf-8")
            if total.endswith("ff"):
                break
        return total

    def end(self):
        self._tcp_socket.close()
        self._udp_socket.close()

class PTZOptics20x(TCPCamera):
    """PTZOptics VISCA control class.

    Tested with USB 20X model.
    """
    # Pan/tilt continuing motion
    _ptContinuousMotion = False
    # Continuous zoom change initiated
    _zContinuous = False

    def __init__(self, host, tcp_port=5678, udp_port=1259):
        """Sony VISCA control class.

        :param host: TCP control host or IP address
        :type host: str
        :param port: TCP control port
        :type port: int
        """
        super(self.__class__, self).__init__(host, tcp_port=tcp_port, udp_port=udp_port)

    def init(self):
        """Initializes camera object by connecting to TCP control socket.

        :return: Camera object.
        :rtype: TCPCamera
        """
        if super(self.__class__, self).init() is None:
            return None
        print("Camera controller initialized")
        self.comm('8101043803FF', 'udp') # Set focus to manual
        return self

    def panTiltOngoing(self):
        return True if self._ptContinuousMotion else False

    def zoomOngoing(self):
        return True if self._zContinuous else False

    def comm(self, com, channel):
        """Sends hexadecimal string to control socket.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
        """
        super(self.__class__, self).command(com, channel)

    @staticmethod
    def multi_replace(text, rep):
        """Replaces multiple parts of a string using regular expressions.

        :param text: Text to be replaced.
        :type text: str
        :param rep: Dictionary of key strings that are replaced with value strings.
        :type rep: dict
        :return: Replaced string.
        :rtype: str
        """
        rep = dict((re.escape(k), v) for k, v in rep.iteritems())
        pattern = re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    def get_zoom_position(self):
        """Retrieves current zoom position.
        Zoom is 0 to 16384

        :return: Zoom distance
        :rtype: int
        """
        self.comm('81090447FF', 'tcp')
        msg = self.read()[4:-2]
        r = ""
        if len(msg) == 8:
            for x in range(1, 9, 2):
                r += msg[x]
            x = int(r, 16)
            return True, x
        return False, -1

    def get_pan_tilt_position(self):
        """Retrieves current pan/tilt position.
        Pan is 0 at home. Right is positive, max 2448. Left ranges from full left 63088 to 65535 before home.
        Tilt is 0 at home. Up is positive, max 1296. Down ranges from fully depressed at 65104 to 65535 before home.

        :return: pan position
        :rtype: int
        :return: tilt position
        :rtype: int
        """
        self.comm('81090612FF', 'tcp')
        msg = self.read()[4:-2]
        r = ""
        if len(msg) == 16:
            for x in range(1, 9, 2):
                r += msg[x]
            pan = int(r, 16)
            r = ""
            for x in range(9, 17, 2):
                r += msg[x]
            tilt = int(r, 16)
            return True, pan, tilt
        return False, -1, -1

    def home(self):
        """Moves camera to home position.

        :return: True if successful, False if not.
        :rtype: bool
        """
        # Since home is not continuing motion, we'll call it a stop
        self._ptContinuousMotion = False
        return self.comm('81010604FF', 'udp')

    def reset(self):
        """Resets camera.

        :return: True if successful, False if not.
        :rtype: bool
        """
        self._ptContinuousMotion = False
        self._zContinuous = False
        return self.comm('81010605FF', 'udp')

    def stop(self):
        """Stops camera movement (pan/tilt).

        :return: True if successful, False if not.
        :rtype: bool
        """
        self._ptContinuousMotion = False
        return self.comm('8101060115150303FF', 'udp')

    def cancel(self):
        """Cancels current command.

        :return: True if successful, False if not.
        :rtype: bool
        """
        self._ptContinuousMotion = False
        self._zContinuous = False
        return self.comm('81010001FF', 'udp')

    def _move(self, string, a1, a2):
        h1 = "%X" % a1
        h1 = '0' + h1 if len(h1) < 2 else h1

        h2 = "%X" % a2
        h2 = '0' + h2 if len(h2) < 2 else h2
        self._ptContinuousMotion = True
        return self.comm(string.replace('VV', h1).replace('WW', h2), 'udp')

    def goto(self, pan, tilt, speed=5):
        """Moves camera to absolute pan and tilt coordinates.

        :param speed: Speed (0-24)
        :param pan: numeric pan position
        :param tilt: numeric tilt position
        :return: True if successful, False if not.
        :rtype: bool
        """
        speed_hex = "%X" % speed
        speed_hex = '0' + speed_hex if len(speed_hex) < 2 else speed_hex

        pan_hex = "%X" % pan
        pan_hex = pan_hex if len(pan_hex) > 3 else ("0" * (4 - len(pan_hex))) + pan_hex
        pan_hex = "0" + "0".join(pan_hex)

        tilt_hex = "%X" % tilt
        tilt_hex = tilt_hex if len(tilt_hex) > 3 else ("0" * (4 - len(tilt_hex))) + tilt_hex
        tilt_hex = "0" + "0".join(tilt_hex)

        s = '81010602VVWWYYYYZZZZFF'.replace(
            'VV', speed_hex).replace(
            'WW', speed_hex).replace(
            'YYYY', pan_hex).replace(
            'ZZZZ', tilt_hex)

        # Not in continuing motion
        self._ptContinuousMotion = False

        return self.comm(s, 'udp')

    def gotoIncremental(self, pan, tilt, speed=5):
        """Moves camera to relative pan and tilt coordinates.

        :param speed: Speed (0-24)
        :param pan: numeric pan adjustment
        :param tilt: numeric tilt adjustment
        :return: True if successful, False if not.
        :rtype: bool
        """
        speed_hex = "%X" % speed
        speed_hex = '0' + speed_hex if len(speed_hex) < 2 else speed_hex

        pan_hex = "%X" % pan
        pan_hex = pan_hex if len(pan_hex) > 3 else ("0" * (4 - len(pan_hex))) + pan_hex
        pan_hex = "0" + "0".join(pan_hex)

        tilt_hex = "%X" % tilt
        tilt_hex = tilt_hex if len(tilt_hex) > 3 else ("0" * (4 - len(tilt_hex))) + tilt_hex
        tilt_hex = "0" + "0".join(tilt_hex)

        s = '81010603VVWWYYYYZZZZFF'.replace(
            'VV', speed_hex).replace(
            'WW', speed_hex).replace(
            'YYYY', pan_hex).replace(
            'ZZZZ', tilt_hex)

        # Not in continuing motion
        self._ptContinuousMotion = False

        return self.comm(s, 'udp')

    def zoomstop(self):
        """Halt the zoom motor

        :return: True on success, False on failure
        :rtype: bool
        """
        s = '8101040700FF'
        self._zContinuous = False
        return self.comm(s, 'udp')

    def zoomin(self, speed=0):
        """Initiate tele zoom at speed range 0-7

        :param speed: zoom speed, 0-7
        :return: True on success, False on failure
        :rtype: bool
        """
        if speed < 0 or speed > 7:
            return False
        s = '810104072pFF'.replace(
            'p', "{0:1s}".format(str(speed)))
        # print("zoomin comm string: " + s)
        self._zContinuous = True
        return self.comm(s, 'udp')

    def zoomout(self, speed=0):
        """Initiate tele zoom at speed range 0-7

        :param speed: zoom speed, 0-7
        :return: True on success, False on failure
        :rtype: bool
        """
        if speed < 0 or speed > 7:
            return False
        s = '810104073pFF'.replace(
            'p', "{0:1s}".format(str(speed)))
        # print("zoomout comm string: " + s)
        self._zContinuous = True
        return self.comm(s, 'udp')

    def zoomto(self, zoom, channel='udp'):
        """Moves camera to absolute zoom setting.

        :param zoom: numeric zoom position
        :return: True if successful, False if not.
        :rtype: bool
        """
        zoom_hex = "%X" % zoom
        zoom_hex = zoom_hex if len(zoom_hex) > 3 else ("0" * (4 - len(zoom_hex))) + zoom_hex
        zoom_hex = "0" + "0".join(zoom_hex)


        s = '81010447pqrsFF'.replace(
            'pqrs', zoom_hex)
        return self.comm(s, channel)

    def left(self, amount=5):
        """Modifies pan speed to left.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        :rtype: bool
        """
        hex_string = "%X" % amount
        hex_string = '0' + hex_string if len(hex_string) < 2 else hex_string
        s = '81010601VVWW0103FF'.replace('VV', hex_string).replace('WW', str(15))
        self._ptContinuousMotion = True
        return self.comm(s, 'udp')

    def right(self, amount=5):
        """Modifies pan speed to right.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hex_string = "%X" % amount
        hex_string = '0' + hex_string if len(hex_string) < 2 else hex_string
        s = '81010601VVWW0203FF'.replace('VV', hex_string).replace('WW', str(15))
        self._ptContinuousMotion = True
        return self.comm(s, 'udp')

    def up(self, amount=5):
        """Modifies tilt speed to up.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hs = "%X" % amount
        hs = '0' + hs if len(hs) < 2 else hs
        s = '81010601VVWW0301FF'.replace('VV', str(15)).replace('WW', hs)
        self._ptContinuousMotion = True
        return self.comm(s, 'udp')

    def down(self, amount=5):
        """Modifies tilt to down.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hs = "%X" % amount
        hs = '0' + hs if len(hs) < 2 else hs
        s = '81010601VVWW0302FF'.replace('VV', str(15)).replace('WW', hs)
        self._ptContinuousMotion = True
        return self.comm(s, 'udp')

    def left_up(self, pan, tilt):
        return self._move('81010601VVWW0101FF', pan, tilt)

    def right_up(self, pan, tilt):
        return self._move('81010601VVWW0201FF', pan, tilt)

    def left_down(self, pan, tilt):
        return self._move('81010601VVWW0102FF', pan, tilt)

    def right_down(self, pan, tilt):
        return self._move('81010601VVWW0202FF', pan, tilt)


class PIDController:
    def __init__(self, kp=1.2, kd=0.1, ki=0.1, T=1/50, omega_c=2*3.14*10):
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

def expand_bbox(x, y, w, h, width=1920, height=1080):
    diff = w - h

    if diff > 0: # Wider image. Increase height
        y -= diff // 2
        h += diff
        if y < 0:
            y = max(0, y)  # Takes care of out of bounds upwards
        if y + h > height: 
            y = height - h # Takes care of out of bounds downwards
        
    elif diff < 0: # Taller image. Increase width
        x -= abs(diff) // 2
        w += abs(diff)
        if x < 0:
            x = max(0, x)  # Takes care of out of bounds from left
        if x + w > width: 
            x = width - w # Takes care of out of bounds from right
            
    return x, y, w, h
    
