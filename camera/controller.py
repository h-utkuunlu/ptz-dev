from time import sleep, time

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

def errors_pt(center, width, height):
    return ((width//2 - center[0])/50, (height//2 - center[1])/30) # Pos. pan error: right. Pos. tilt error: down

def limit(val, max):
    return max if abs(val) > max else abs(int(val))
