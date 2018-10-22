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
    int opt;
    po::options_description desc("Allowed options");
    desc.add_options()
    ('H,host', po::value<vector<string>>(&opt)->default_value('192.168.1.40'), "Host address for the camera for PTZ control")
    ('p,port', po::value<vector<int>>(&opt)->default_value(5678), "Port for TCP comms PTZ control")
    ('d,dev', po::value<vector<int>>(&opt)->default_value(0), 'Camera USB device location for OpenCV')
    ('c,count', po::value<vector<int>>(&opt)->default_value(600), 'Number of frames to run for non-GUI mode')
    ('g,gui', po::value<vector<bool>>(&opt)->default_value(true),'GUI mode');

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