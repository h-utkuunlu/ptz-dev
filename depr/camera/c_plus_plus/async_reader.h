#ifndef ASYNC_READER_H
#define ASYNC_READER_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <unistd.h>
#include "frame_counter.h"

namespace Capstone {

    class WeightedFramerateCounter;

    class CameraReaderAsync {
    private:
        cv::VideoCapture cvCamera;
        cv::Mat frame;
        bool lastFrameRead;
        bool stopRequested;
        bool validFrame;
        WeightedFramerateCounter fps;
        std::mutex mtx;
        std::vector<std::thread> th;

    public:
        CameraReaderAsync (const int usbdevnum, const int width, const int height);

        void readAsync ();

        void stop();

        void release();

        cv::Mat read();

        cv::Mat readLastFrame();
    };


    class Camera : public CameraReaderAsync {
    private:
        int panPos;
        int tiltPos;
        int zoomPos;
        int badCountPTZ;

    public:
        Camera (const int usbdevnum, const double width, const double height);

        bool lostPTZfeed();

        void updatePTZ();
    };
}


#endif
