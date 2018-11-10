#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <unistd.h>
#include "PID_controller.h"
#include "camera_controls.h"
#include "async_reader.h"

using namespace cv;
using namespace std;
using namespace std::chrono;
using namespace Capstone;

cv::Rect find_object(cv::Mat tempFrame);
std::tuple<float, float> errors(std::tuple<int, int> center, float width, float height);
size_t limit(float val, size_t max);
std::tuple<int, int> control(const std::tuple<float, float>& errors, PIDController& panPID, PIDController& tiltPID, PTZOptics20x& ptz);
std::tuple<int, int> computeCenter(cv::Rect& boundingBox);

int main(int argc, char** argv )
{
    // Command-line arguments
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()  // returns special proxy object that defines operator()
        ("help,h", "produce help message")
        ("host,H", po::value<string>()->default_value("192.168.1.40"), "Host address for the camera for PTZ control")
        ("port,p", po::value<string>()->default_value("1259"), "Port for TCP comms PTZ control")
        ("dev,d", po::value<int>()->default_value(0), "Camera USB device location for OpenCV")
        ("count,c", po::value<int>()->default_value(600), "Number of frames to run for non-GUI mode")
        ("gui,g", po::value<bool>()->default_value(true),"GUI mode");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    bool guiMode = vm["gui"].as<bool>();

    string host = vm["host"].as<string>();
    string port = vm["port"].as<string>();

    std::cout << host << " " << port << '\n';
    PTZOptics20x ptz(host, port);
    ptz.connect();

    int width = 1280;
    int height = 720;
    int device = vm["dev"].as<int>();
    Camera source(device, width, height);

    if (guiMode) {
        cv::namedWindow("cam", WINDOW_AUTOSIZE);
    }

    PIDController panPID(1.4, 0.0, 0.0, 1.0/50.0, 2.0*3.14*10.0);
    PIDController tiltPID(1.4, 0.0, 0.0, 1.0/50.0, 2.0*3.14*10.0);

    milliseconds start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    milliseconds end;

    while (true) {
        cv::Mat frame = source.read();
        if (frame.empty()) { continue; }

        cv::Rect boundingBox = find_object(frame);

        if (!boundingBox.area()) {
            if (guiMode) {
                cv::imshow("cam", frame);
                char key = waitKey(1);
                if (key == 'q') { break; }
            }
            continue;
        }

        if (guiMode) {
            cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 255));
            cv::imshow("cam", frame);
            char key = waitKey(1);
            if (key == 'q') { break; }
        }

        std::tuple<int, int> center = computeCenter(boundingBox);

        std::tuple<int, int> speeds = control(errors(center, width, height), panPID, tiltPID, ptz);

        end = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        float dur = (end - start).count();  // duration in milliseconds
        int FPS = 1/(dur/1000);
        std::cout << "Approx. Frame Rate: " << FPS << '\n';
        start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    }

    source.stop();
    source.release();
    if (guiMode) {
        cv::destroyAllWindows();
    }
    ptz.stop();
    usleep(1000000);
    std::cout << "Program completed!" << '\n';
}

cv::Rect find_object(cv::Mat tempFrame) {
    cv::Mat frame = tempFrame.clone();
    int low_H = 100;
    int low_S = 50;
    int low_V = 50;
    int high_H = 160;
    int high_S = 255;
    int high_V = 255;

    cv::blur(frame, frame, cv::Size(11,11));
    cv::cvtColor(frame, frame, COLOR_BGR2HSV);
    cv::inRange(frame, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), frame);
    cv::erode(frame, frame, cv::Mat()); cv::erode(frame, frame, cv::Mat());
    cv::dilate(frame, frame, cv::Mat()); cv::dilate(frame, frame, cv::Mat());

    vector<vector<cv::Point>> contours; // Vector for storing contour
    cv::findContours(frame, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    if (contours.size() == 0) { return cv::Rect(); }
    int largest_area = 0;
    size_t largest_contour_index = 0;
    for(size_t index = 0; index < contours.size(); index++) {
        double area = cv::contourArea(contours[index], false); //  Find the area of contour
        if (area > largest_area) {
            largest_area = area;
            // Store the index of largest contour
            largest_contour_index = index;
        }
    }
    // Find the bounding rectangle for biggest contour

    return boundingRect(contours[largest_contour_index]);
}

std::tuple<float, float> errors(std::tuple<int, int> center, float width, float height) {
    float centerX = std::get<0>(center);
    float centerY = std::get<1>(center);
    float pan_error = (width/2 - centerX)/50;
    float tilt_error = (height/2 - centerY)/30;
    return std::make_tuple(pan_error, tilt_error);
}

size_t limit(float val, size_t max) {
    if (abs(val) > max) { return max; }
    else { return (size_t)abs(val); }
}

std::tuple<int, int>
control(const std::tuple<float, float>& errors, PIDController& panPID, PIDController& tiltPID, PTZOptics20x& ptz) {

    unsigned int dur = 10;

    // std::cout << "Error: " << std::get<0>(errors) << ", " << std::get<1>(errors) << '\n';

    float pan_command = panPID.compute(std::get<0>(errors));
    float tilt_command = tiltPID.compute(std::get<1>(errors));

    // std::cout << "pan_command: " << pan_command << '\n';
    // std::cout << "tilt_command: " << tilt_command << '\n';

    size_t pan_speed = limit(pan_command, 24);
    size_t tilt_speed = limit(tilt_command, 20);

    // std::cout << "Pan Speed: " << pan_speed << '\n';
    // std::cout << "Tilt Speed: " << tilt_speed << '\n';

    if ((pan_speed == 0) and (tilt_speed == 0)) {
        ptz.leftUp(1, 1);
        usleep(dur);
    } else if (pan_speed == 0) {
        if (tilt_command == abs(tilt_command)) {
            ptz.up(tilt_speed);
            usleep(dur);
        } else {
            ptz.down(tilt_speed);
            usleep(dur);
        }
    } else if (tilt_speed == 0) {
        if (pan_command == abs(pan_command)) {
            ptz.left(pan_speed);
            usleep(dur);
        } else {
            ptz.right(pan_speed);
            usleep(dur);
        }
    } else if ((abs(pan_command) == pan_command) && (abs(tilt_command) == tilt_command)) {
        ptz.leftUp(pan_speed, tilt_speed);
        usleep(dur);
    } else if ((abs(pan_command) == pan_command) && (abs(tilt_command) != tilt_command)) {
        ptz.leftDown(pan_speed, tilt_speed);
        usleep(dur);
    } else if ((abs(pan_command) != pan_command) && (abs(tilt_command) == tilt_command)) {
        ptz.rightUp(pan_speed, tilt_speed);
        usleep(dur);
    } else if ((abs(pan_command) != pan_command) && (abs(tilt_command) != tilt_command)) {
        ptz.rightDown(pan_speed, tilt_speed);
        usleep(dur);
    }
    return std::make_tuple(pan_speed, tilt_speed);
}

std::tuple<int, int> computeCenter(cv::Rect& boundingBox) {
    int centerX = boundingBox.x + (boundingBox.width/2);
    int centerY = boundingBox.y + (boundingBox.height/2);
    return std::make_tuple(centerX, centerY);
}
