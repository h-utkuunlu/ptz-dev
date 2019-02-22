#include "async_reader.h"
#include "frame_counter.h"

namespace Capstone {

    CameraReaderAsync::CameraReaderAsync (const int usbdevnum, const int width, const int height) : cvCamera(usbdevnum) {
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

    void CameraReaderAsync::readAsync (){
        while (true) {
            if (stopRequested) { return; }
            cv::Mat tempframe;
            validFrame = cvCamera.read(tempframe);
            if (validFrame) {
                std::lock_guard<std::mutex> lk(mtx);
                fps.tick();
                frame = tempframe.clone();
                lastFrameRead = false;
            }

        }
    }

    void CameraReaderAsync::stop() {
        stopRequested = true;
    }

    void CameraReaderAsync::release() {
        cvCamera.release();
    }

    cv::Mat CameraReaderAsync::read() {
        std::lock_guard<std::mutex> lk(mtx);
        if (!lastFrameRead) {
            lastFrameRead = true;
            return frame;
        }
        return cv::Mat();
    }

    cv::Mat CameraReaderAsync::readLastFrame() {
        return frame;
    }

    Camera::Camera (const int usbdevnum, const double width, const double height)
    : CameraReaderAsync(usbdevnum, width, height) {
        panPos = 0;
        tiltPos = 0;
        zoomPos = 0;
        badCountPTZ = 0;
    };

    bool Camera::lostPTZfeed() {
        if (badCountPTZ < 5) { return false; }
        else { return true; }
    }

    void Camera::updatePTZ() {
        badCountPTZ++;
    }
}
