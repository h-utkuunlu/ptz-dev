from threading import Thread
from threading import Lock
from frame_counter import WeightedFramerateCounter
import cv2

class CameraReaderAsync:
    def __init__(self, videoSource):
        self.__lock = Lock()
        self.__source = videoSource
        self.Start()
        
    def __ReadAsync(self):
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

    def Start(self):
        self.__lastFrameRead = False
        self.__frame = None
        self.__stopRequested = False
        self.__validFrame = False
        self.fps = WeightedFramerateCounter()
        Thread(target=self.__ReadAsync).start()
        
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

    # Return the last frame read even if it has been retrieved before.
    # Will return None if we never read a valid frame from the source.
    def ReadLastFrame(self):
        return self.__frame

class Camera():
    cvcamera = None
    cvreader = None
    width = 0
    height = 0
    panPos = 0
    tiltPos = 0
    zoomPos = -1
    _badPTZcount = 0
    
    def __init__(self, usbdevnum, width, height):
        # Start by establishing control connection
        #pysca.connect(cfg['socket'])

        # Open video stream as CV camera
        self.cvcamera = cv2.VideoCapture(usbdevnum)
        self.width = width
        self.height = height
        self.cvcamera.set(3, width)
        self.cvcamera.set(4, height)

        self.cvreader = CameraReaderAsync(self.cvcamera)
    
    def lostPTZfeed(self):
        return True
        # return False if self._badPTZcount < 5 else True
            
    def updatePTZ(self):
        self._badPTZcount += 1

