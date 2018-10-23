#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main(int argc, char** argv )
{
    // Command-line arguments
    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()  // returns special proxy object that defines operator()
        ("help,h", "produce help message")
        ("host,H", po::value<string>()->default_value("192.168.1.40"), "Host address for the camera for PTZ control")
        ("port,p", po::value<int>()->default_value(5678), "Port for TCP comms PTZ control")
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
    if (vm.count("host")) {
        cout << vm["host"].as<string>() << endl;
    }
}
