#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <exception>


namespace ba = boost::asio;
using ba::ip::udp;

class UDPCamera {
private:
    const std::string udp_host;
    const std::string udp_port;
    ba::io_service io;
    udp::socket socket;
    udp::endpoint receiver_endpoint;

public:
    /*
    PTZOptics VISCA control class.

    :param host: UDP control host.
    :type host: str
    :param port: UDP control port.
    :type port: str
    */
    UDPCamera (const std::string& host, const std::string& port)
        : udp_host(host), udp_port(port), io(), socket(io) { }

    void command() {
        try {
            uint8_t foo[] = {0x81, 0x01, 0x06, 0x04, 0xFF};
    		socket.send_to(ba::buffer(foo), receiver_endpoint);
        } catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }

    /*
    Initializes camera object by establishing TCP control session.

    :return: Camera object.
    :rtype: TCPCamera
    */
    void connect() {
        try {
            std::cout << "Connecting to camera..." << std::endl;
            udp::resolver resolver(io);
    		udp::resolver::query query(udp::v4(), udp_host, udp_port);
    		receiver_endpoint = *resolver.resolve(query);
    		socket.open(udp::v4());
            std::cout << "Connected to camera successfully" << std::endl;
        } catch (std::exception& e){
            std::cerr << e.what() << std::endl;
        }
    }

};

int main(int argc, char* argv[]) {
    const std::string PORT("1259");
    const std::string HOSTNAME("192.168.1.40");
    UDPCamera myCam(HOSTNAME, PORT);
    myCam.connect();
    myCam.command();
}

//
// class PTZOptics20x {
// private:
//     _socket = None
//     _host = None
//     _tcp_port = None
//
// public:
//     PTZOptics20x (arguments);
// };
