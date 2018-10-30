#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>


using namespace cv;

class CameraReaderAsync {
private:
    VideoCapture source;
    cv::Mat frame;
    bool lastFrameRead;
    bool stopRequested;
    bool validFrame;
    // WeightedFramerateCounter fps;
    std::mutex mtx;
    std::vector<std::thread> th;

public:
    CameraReaderAsync (VideoCapture& videoSource) : source(videoSource) {
        if (!source.isOpened()) {
            std::cout << "Problem connecting to cam" << std::endl;
            exit(1);
        }
        lastFrameRead = false;
        stopRequested = false;
        validFrame = false;
        th.push_back(std::thread(&CameraReaderAsync::readAsync, this));
    }

    void readAsync (){
        while (true) {
            if (stopRequested) { return; }
            Mat tempframe;
            validFrame = source.read(tempframe);
            if (validFrame) {
                std::lock_guard<std::mutex> lk(mtx);
                // fps.tick();
                frame = tempframe.clone();
                lastFrameRead = false;
            }

        }
    }

    void stop() {
        stopRequested = true;
    }

    cv::Mat read() {
        std::lock_guard<std::mutex> lk(mtx);
        if (!lastFrameRead) {
            lastFrameRead = true;
            return cv::Mat(frame);
        }
        return cv::Mat();
    }

    cv::Mat readLastFrame() {
        return frame;
    }
};


class Camera {
private:
    /* data */

public:
    Camera ();
};


int main(int argc, char const *argv[]) {
    /* code */
    return 0;
}
