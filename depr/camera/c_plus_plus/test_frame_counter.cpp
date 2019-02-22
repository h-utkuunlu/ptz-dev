#include "frame_counter.h"
#include <unistd.h>
#include <iostream>

using namespace std;
using namespace Capstone;

int main(int argc, char const *argv[]) {
    WeightedFramerateCounter fps;
    size_t count = 100;
    for (size_t i = 0; i < count; i++) {
        fps.tick();
        usleep(1000000);
        fps.tick();
        std::cout << "FPS: " << fps.getFrameRate() << '\n';
    }
    return 0;
}
