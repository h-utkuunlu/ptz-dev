#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/algorithm/hex.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include <unistd.h>


namespace ba = boost::asio;
using boost::asio::ip::udp;

class UDPCamera {
private:
    const std::string udp_host;
    const std::string udp_port;
    ba::io_service io;
    udp::socket socket;
    udp::endpoint receiver_endpoint;
    boost::array<char, 16> recv_buffer;

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

    /*
    Sends hexadecimal string to TCP control socket.

        :param com: Command string. Hexadecimal format.
        :type com: str
        :return: Success.
        :rtype: bool
    */
    bool command(const std::string& commStr) {
        std::vector<uint8_t> comm;
        boost::algorithm::unhex(commStr.begin(), commStr.end(), std::back_inserter(comm));
        try {
    		socket.send_to(ba::buffer(comm), receiver_endpoint);
            return true;
        } catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
            return false;
        }
    }

    std::string read() {
        std::string res;
        socket.receive_from(
            boost::asio::buffer(recv_buffer),
            receiver_endpoint);
        boost::algorithm::hex(recv_buffer.begin(), recv_buffer.end(), back_inserter(res));
        return res;
    }

    void end() {
        boost::system::error_code ec;
        socket.shutdown(udp::socket::shutdown_both, ec);
        socket.close();
    }

};

class PTZOptics20x : public UDPCamera {
private:
    bool ptContinuousMotion = false;
    bool zContinuous = false;

public:
    PTZOptics20x(const std::string& host, const std::string& port)
        : UDPCamera(host, port) {}

    void connect() {
        UDPCamera::connect();
    }

    bool panTiltOngoing() {
        return ptContinuousMotion;
    }

    bool zoomOnGoing() {
        return zContinuous;
    }

    /*
    Retrieves current pan/tilt position.
    Pan is 0 at home. Right is positive, max 2448. Left ranges from full left 63088 to 65555 before home.
    Tilt is 0 at home. Up is positive, max 1296. Down ranges from fully depressed at 65104 to 65555 before home.

        :return: pan position
        :rtype: int
        :return: tilt position
        :rtype: int
    */
    std::tuple<unsigned int, unsigned int> getPanTiltPosition() {
        command("81090612FF");
        std::string response = read();
        std::string r;
        unsigned int pan;
        unsigned int tilt;
        if (response.length() == 16) {
            for (size_t index = 1; index < 9; index += 2) {
                r += response[index];
            }
            pan = std::stoul(r, nullptr, 16);
            r.clear();
            for (size_t index = 9; index < 16; index += 2) {
                r += response[index];
            }
            tilt = std::stoul(r, nullptr, 16);
            return std::make_tuple(pan, tilt);
        }
        return std::make_tuple(1, -1);
    }

    bool home() {
        ptContinuousMotion = false;
        return command("81010604FF");
    }



};

int main(int argc, char* argv[]) {
    const std::string PORT("1259");
    const std::string HOSTNAME("192.168.1.40");
    UDPCamera myCam(HOSTNAME, PORT);
    myCam.connect();
    //std::vector<uint8_t> comm({0x81, 0x01, 0x06, 0x04, 0xFF}); // Home position
    //std::vector<uint8_t> comm({0x81, 0x09, 0x06, 0x23, 0xFF}); // Video format
    //std::vector<uint8_t> comm({0x81, 0x09, 0x06, 0x12, 0xFF}); // Inquire position
    std::vector<uint8_t> comm({0x81, 0x01, 0x06, 0x02, 0x18, 0x14, 0x0F, 0x0A, 0x0E, 0x01, 0x00, 0x03, 0x0E, 0x08, 0xFF});


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
