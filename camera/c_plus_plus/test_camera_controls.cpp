#include "camera_controls.h"

using namespace Capstone;

int main(int argc, char* argv[]) {
    const std::string PORT("1259");
    const std::string HOSTNAME("192.168.1.40");
    PTZOptics20x myCam(HOSTNAME, PORT);
    myCam.connect();
    // std::string comm = 81010604FF; // Home position
    // std::string comm = 81090623FF; // Video format
    // std::string comm = 81090612FF; // Inquire position
    // std::string comm = 8101060218140F0A0E0100030E08FF; // absolute position
    std::string comm = "81010604FF";
    myCam.command(comm);
    usleep(2000000);
    myCam.right(24);
    usleep(1000000);
    myCam.left(25);
    usleep(2000000);
    myCam.command(comm);

}
