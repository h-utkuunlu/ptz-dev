#ifndef FRAME_COUNTER_H
#define FRAME_COUNTER_H

#include <chrono>

namespace Capstone {
    class WeightedFramerateCounter {
    private:
        float smoothing = 0.95;
        std::chrono::duration<double> start = std::chrono::duration<double>::zero();
        float frameRate = 0.0;

    public:
        void startTime();

        void tick();

        float getFrameRate();
    };
}

#endif
