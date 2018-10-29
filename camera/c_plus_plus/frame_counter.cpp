#include <chrono>

using namespace std::chrono;

class WeightedFramerateCounter {
private:
    float smoothing = 0.95;
    duration<double> start = duration<double>::zero();
    float frameRate = 0.0;


public:
    void startTime() {
        start = high_resolution_clock::now().time_since_epoch();
        frameRate = 0.0;
    }

    void tick() {
        duration<double> timeNow = high_resolution_clock::now().time_since_epoch();
        if (start == duration<double>::zero()) {
            start = timeNow;
            return;
        }
        double elapsed = 1.0 / (timeNow - start).count();
        start = timeNow;
        frameRate = (frameRate * smoothing) + (elapsed * (1.0 * smoothing));
    }

    float getFrameRate() {
        return frameRate;
    }
};

int main(int argc, char const *argv[]) {

    return 0;
}
