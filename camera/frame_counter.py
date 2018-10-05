import time

class WeightedFramerateCounter:
    smoothing = 0.95
    startTime = 0
    framerate = 0

    def start(self):
        self.startTime = time.time()
        self.framerate = 0
        
    def tick(self):
        timeNow = time.time()
        if self.startTime == 0:
            self.startTime = timeNow
            return
        elapsed = 1.0 / (timeNow - self.startTime)
        self.startTime = timeNow
        self.framerate = (self.framerate * self.smoothing) + (elapsed * (1.0 - self.smoothing))

    def getFramerate(self):
        return self.framerate 
