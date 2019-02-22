#include "PID_controller.h"

namespace Capstone {

    float PIDController::df_bil(float diff_1, float err, float err_1, float T) const {
        return -diff_1 + (err - err_1) * (2 / T);
    }

    float PIDController::filt_bil(float filt_1, float diff, float diff_1, float T, float omega_c) const {
        return ((2 - T*omega_c)/(2 + T*omega_c))*filt_1 + T*omega_c*(diff + diff_1)/(2+T*omega_c);
    }

    float PIDController::integ_bil(float integ_1, float err, float err_1, float T) const {
        return integ_1 + (err + err_1)*(T/2);
    }

    PIDController::PIDController(float kp, float kd, float ki, float T, float omega_c)
    : past({{"err", 0.0}, {"diff", 0.0}, {"filt", 0.0}, {"integ", 0.0}}) {
        this->kp = kp;
        this->kd = kd;
        this->ki = ki;
        this->T = T;
        this->omega_c = omega_c;
        minval = -24;
        maxval = 24;
    }

    float PIDController::compute(float err) {
        float diff = df_bil(past["diff"], err, past["err"], T);
        float filt = filt_bil(past["filt"], diff, past["diff"], T, omega_c);
        float integ = integ_bil(past["integ"], err, past["err"], T);

        // std::cout << "diff: " << diff << " , filt: " << filt << ", integ: " << integ << '\n';

        float pid_out = kp*err + kd*filt + ki*integ;
        // std::cout << "Compute pid_out: " << pid_out << '\n';

        past["err"] = err;
        past["diff"] = diff;
        past["filt"] = filt;

        if ((pid_out > minval) && (pid_out < maxval)) {
            past["integ"] = integ;
        }

        return pid_out;
    }

}
