#include "frame_counter.h"

namespace Capstone {

    void WeightedFramerateCounter::startTime() {
        start = std::chrono::high_resolution_clock::now().time_since_epoch();
        frameRate = 0.0;
    }

    void WeightedFramerateCounter::tick() {
        std::chrono::duration<double> timeNow = std::chrono::high_resolution_clock::now().time_since_epoch();
        if (start == std::chrono::duration<double>::zero()) {
            start = timeNow;
            return;
        }
        double elapsed = 1.0 / (timeNow - start).count();
        start = timeNow;
        frameRate = (frameRate * smoothing) + (elapsed * (1.0 * smoothing));
    }

    float WeightedFramerateCounter::getFrameRate() {
        return frameRate;
    }
}
