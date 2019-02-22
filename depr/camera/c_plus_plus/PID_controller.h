#ifndef PID_CONTROLLER_H
#define PID_CONTROLLER_H

#include <string>
#include <unordered_map>
#include <iostream>

namespace Capstone {
    class PIDController {
    private:
        std::unordered_map<std::string, float> past;
        float kp;
        float kd;
        float ki;
        float T;
        float omega_c;
        short int minval;
        short int maxval;

        float df_bil(float diff_1, float err, float err_1, float T) const;

        float filt_bil(float filt_1, float diff, float diff_1, float T, float omega_c) const;

        float integ_bil(float integ_1, float err, float err_1, float T) const;

    public:
        PIDController(float kp, float kd, float ki, float T, float omega_c);

        float compute(float err);
    };
}

#endif
