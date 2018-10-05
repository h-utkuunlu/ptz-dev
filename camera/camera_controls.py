from __future__ import print_function
import re
import binascii
import socket
import fcntl
import os, sys
import errno
import numpy as np

class TCPCamera(object):
    _socket = None
    _tcp_host = None
    _tcp_port = None

    def __init__(self, host, port):
        """PTZOptics VISCA control class.

        :param host: TCP control host.
        :type host: str
        :param port: TCP control port.
        :type port: int
        """
        self._tcp_host = host
        self._tcp_port = port

    def init(self):
        """Initializes camera object by establishing TCP control session.

        :return: Camera object.
        :rtype: TCPCamera
        """
        print("Connecting to camera...")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(0.6)
        try:
            self._socket.connect((self._tcp_host, self._tcp_port))
        except:
            print("Could not connect to camera on control channel")
            return None
        print("Camera connected")
        self._socket.settimeout(0.2)
        return self

    def command(self, com):
        """Sends hexadecimal string to TCP control socket.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
        """
        try:
            self._socket.send(binascii.unhexlify(com))
            return True
        except Exception as e:
            print(com, e)
            return False

    def read(self, amount=1):
        total = ""
        while True:
            try:
                msg = binascii.hexlify(self._socket.recv(amount))
            except socket.timeout:
                print("No data from camera socket")
                break
            except socket.error:
                print("Camera socket read error.")
                break
            total = total + msg
            if msg == "ff":
                break
        return total

    def end(self):
        self._socket.close()


class PTZOptics20x(TCPCamera):
    """PTZOptics VISCA control class.

    Tested with USB 20X model.
    """
    # Pan/tilt continuing motion
    _ptContinuousMotion = False
    # Continuous zoom change initiated
    _zContinuous = False
    
    def __init__(self, host, port):
        """Sony VISCA control class.

        :param host: TCP control host or IP address
        :type host: str
        :param port: TCP control port
        :type port: int
        """
        super(self.__class__, self).__init__(host=host, port=port)

    def init(self):
        """Initializes camera object by connecting to TCP control socket.

        :return: Camera object.
        :rtype: TCPCamera
        """
        if super(self.__class__, self).init() is None:
            return None
        print("Camera controller initialized")
        return self
        
    def panTiltOngoing(self):
        return True if self._ptContinuousMotion else False
        
    def zoomOngoing(self):
        return True if self._zContinuous else False

    def comm(self, com):
        """Sends hexadecimal string to control socket.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
        """
        super(self.__class__, self).command(com)

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
        self.comm('81090447FF')
        msg = self.read()[4:-2]
        r = ""
        if len(msg) == 8:
            for x in range(1, 9, 2):
                r += msg[x]
            x = int(r, 16)
            return x
        return -1

    def get_pan_tilt_position(self):
        """Retrieves current pan/tilt position.
        Pan is 0 at home. Right is positive, max 2448. Left ranges from full left 63088 to 65555 before home.
        Tilt is 0 at home. Up is positive, max 1296. Down ranges from fully depressed at 65104 to 65555 before home.

        :return: pan position
        :rtype: int
        :return: tilt position
        :rtype: int
        """
        self.comm('81090612FF')
        msg = self.read()[4:-2]
        r = ""
        #~ print "pan tilt msg: " + msg
        if len(msg) == 16:
            for x in range(1, 9, 2):
                r += msg[x]
            pan = int(r, 16)
            r = ""
            for x in range(9, 17, 2):
                r += msg[x]
            tilt = int(r, 16)
            return pan, tilt
        return -1,-1

    def home(self):
        """Moves camera to home position.

        :return: True if successful, False if not.
        :rtype: bool
        """
        # Since home is not continuing motion, we'll call it a stop
        self._ptContinuousMotion = False
        return self.comm('81010604FF')

    def reset(self):
        """Resets camera.

        :return: True if successful, False if not.
        :rtype: bool
        """
        self._ptContinuousMotion = False
        self._zContinuous = False
        return self.comm('81010605FF')

    def stop(self):
        """Stops camera movement (pan/tilt).

        :return: True if successful, False if not.
        :rtype: bool
        """
        self._ptContinuousMotion = False
        return self.comm('8101060115150303FF')

    def cancel(self):
        """Cancels current command.

        :return: True if successful, False if not.
        :rtype: bool
        """
        self._ptContinuousMotion = False
        self._zContinuous = False
        return self.comm('81010001FF')

    def _move(self, string, a1, a2):
        h1 = "%X" % a1
        h1 = '0' + h1 if len(h1) < 2 else h1

        h2 = "%X" % a2
        h2 = '0' + h2 if len(h2) < 2 else h2
        self._ptContinuousMotion = True
        return self.comm(string.replace('VV', h1).replace('WW', h2))

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
        
        return self.comm(s)

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
        
        return self.comm(s)
    
    def zoomstop(self):
        """Halt the zoom motor
        
        :return: True on success, False on failure
        :rtype: bool
        """
        s = '8101040700FF'
        self._zContinuous = False
        return self.comm(s)
        
    def zoomin(self, speed=0):
        """Initiate tele zoom at speed range 0-7
        
        :param speed: zoom speed, 0-7
        :return: True on success, False on failure
        :rtype: bool
        """
        if -1 < speed > 7:
            return False
        s = '810104072pFF'.replace(
            'p', "{0:1s}".format(str(speed)))
        print("zoomin comm string: " + s)
        self._zContinuous = True
        return self.comm(s)
            
    def zoomto(self, zoom):
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
        return self.comm(s)

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
        return self.comm(s)

    def right(self, amount=5):
        """Modifies pan speed to right.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hex_string = "%X" % amount
        hex_string = '0' + hex_string if len(hex_string) < 2 else hex_string
        s = '81010601VVWW0203FF'.replace('VV', hex_string).replace('WW', str(15))
        self._ptContinuousMotion = True
        return self.comm(s)

    def up(self, amount=5):
        """Modifies tilt speed to up.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hs = "%X" % amount
        hs = '0' + hs if len(hs) < 2 else hs
        s = '81010601VVWW0301FF'.replace('VV', str(15)).replace('WW', hs)
        self._ptContinuousMotion = True
        return self.comm(s)

    def down(self, amount=5):
        """Modifies tilt to down.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        """
        hs = "%X" % amount
        hs = '0' + hs if len(hs) < 2 else hs
        s = '81010601VVWW0302FF'.replace('VV', str(15)).replace('WW', hs)
        self._ptContinuousMotion = True
        return self.comm(s)

    def left_up(self, pan, tilt):
        return self._move('81010601VVWW0101FF', pan, tilt)

    def right_up(self, pan, tilt):
        return self._move('81010601VVWW0201FF', pan, tilt)

    def left_down(self, pan, tilt):
        return self._move('81010601VVWW0102FF', pan, tilt)

    def right_down(self, pan, tilt):
        return self._move('81010601VVWW0202FF', pan, tilt)
