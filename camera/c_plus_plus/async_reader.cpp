#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <unistd.h>


using namespace cv;

class CameraReaderAsync {
private:
    VideoCapture cvCamera;
    cv::Mat frame;
    bool lastFrameRead;
    bool stopRequested;
    bool validFrame;
    // WeightedFramerateCounter fps;
    std::mutex mtx;
    std::vector<std::thread> th;

public:
    CameraReaderAsync (const int usbdevnum, const int width, const int height) : cvCamera(usbdevnum) {
        if (!cvCamera.isOpened()) {
            std::cout << "Problem with Camera USB device location: " << usbdevnum << std::endl;
            exit(1);
        }
        cvCamera.set(3, width);
        cvCamera.set(4, height);
        lastFrameRead = false;
        stopRequested = false;
        validFrame = false;
        th.push_back(std::thread(&CameraReaderAsync::readAsync, this));
    }

    void readAsync (){
        while (true) {
            if (stopRequested) { return; }
            Mat tempframe;
            validFrame = cvCamera.read(tempframe);
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

    void release() {
        cvCamera.release();
    }

    cv::Mat read() {
        std::lock_guard<std::mutex> lk(mtx);
        if (!lastFrameRead) {
            lastFrameRead = true;
            return frame;
        }
        return cv::Mat();
    }

    cv::Mat readLastFrame() {
        return frame;
    }
};


class Camera : public CameraReaderAsync {
private:
    int panPos;
    int tiltPos;
    int zoomPos;
    int badCountPTZ;

public:
    Camera (const int usbdevnum, const double width, const double height)
        : CameraReaderAsync(usbdevnum, width, height) {
        panPos = 0;
        tiltPos = 0;
        zoomPos = 0;
        badCountPTZ = 0;
    };

    bool lostPTZfeed() {
        if (badCountPTZ < 5) { return false; }
        else { return true; }
    }

    void updatePTZ() {
        badCountPTZ++;
    }
};


int main(int argc, char const *argv[]) {
    const int width = 1280;
    const int height = 720;
    std::cout << "Connecting..." << '\n';
    Camera cam(1, width, height);
    std::cout << "Connected to Camera" << '\n';
    while (true) {
        Mat frame;
        std::cout << "Reading frame" << '\n';
        frame = cam.read();
        std::cout << "Finished reading frame" << '\n';
        namedWindow("Gray Image", WINDOW_AUTOSIZE );
        std::cout << "Displaying..." << '\n';
        if (!frame.empty()) {
            imshow("Gray Image", frame);
        }
        std::cout << "Finished Displaying" << '\n';
        char key = waitKey(1);
        if (key == 'q') { break; }
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
