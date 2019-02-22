#include "async_reader.h"
#include <iostream>
#include <unistd.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace Capstone;
using namespace cv;

int main(int argc, char const *argv[]) {
    const int width = 1280;
    const int height = 720;
    std::cout << "Connecting..." << '\n';
    Camera cam(0, width, height);
    std::cout << "Connected to Camera" << '\n';
    namedWindow("Gray Image", WINDOW_AUTOSIZE );
    Mat frame;
    milliseconds start;
    milliseconds end;
    milliseconds dur;
    start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    while (true) {
        frame = cam.read();
        if (!frame.empty()) {
            imshow("Gray Image", frame);
            end = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
            dur = end - start;
            std::cout << dur.count() << '\n';
            start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());

        }


        char key = waitKey(1);
        if (key == 'q') { break; }

        // std::cout << "FPS ------- " << 1.0/(double)dur << '\n';
    }
    std::cout << "Closing..." << '\n';
    cam.stop();
    usleep(1000000);
    std::cout << "Requested Stop" << '\n';
    cam.release();
    usleep(1000000);
    std::cout << "Released Camera" << '\n';

    return 0;
}
