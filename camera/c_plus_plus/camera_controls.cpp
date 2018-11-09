#include <opencv2/opencv.hpp>
#include <boost/asio.hpp>
#include <boost/array.hpp>
#include <boost/algorithm/hex.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <exception>
#include <sstream>
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

    bool move(std::string& comm, const int pan, const int tilt) {
        std::stringstream stream;
        stream << std::hex << pan;
        std::string h1( stream.str() );

        stream.str(std::string());
        stream << std::hex << tilt;
        std::string h2( stream.str() );

        if (h1.length() < 2) { h1.insert(0, "0"); }
        if (h2.length() < 2) { h2.insert(0, "0"); }
        ptContinuousMotion = true;
        return command(comm.replace(8, 2, h1).replace(10, 2, h2));
    }

    void formatCommand(std::string& comm, const int pan,const int tilt, const size_t speed) {
        std::stringstream stream;
        stream << std::hex << speed;
        std::string speed_hex( stream.str() );
        if (speed_hex.length() < 2) { speed_hex.insert(0, "0"); }

        // Format pan variable
        stream.str(std::string());
        stream << std::hex << pan;
        std::string pan_hex( stream.str() );
        if (pan_hex.length() <= 3) {
            std::string zeros(4-pan_hex.length(), '0');
            pan_hex = zeros + pan_hex;
        }
        stream.str(std::string());
        for (size_t index = 0; index < pan_hex.length(); index++) {
            stream << "0" << pan_hex[index];
        }
        pan_hex = stream.str();

        // Format tilt variable
        stream.str(std::string());
        stream << std::hex << tilt;
        std::string tilt_hex( stream.str() );
        if (tilt_hex.length() <= 3) {
            std::string zeros(4-tilt_hex.length(), '0');
            tilt_hex = zeros + tilt_hex;
        }
        stream.str(std::string());
        for (size_t index = 0; index < tilt_hex.length(); index++) {
            stream << "0" << tilt_hex[index];
        }
        tilt_hex = stream.str();

        comm.replace(
            8, 2, speed_hex).replace(
            10, 2, speed_hex).replace(
            12, 4, pan_hex).replace(
            16, 4, tilt_hex);
    }

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
    std::tuple<int, int> getPanTiltPosition() {
        command("81090612FF");
        std::string response = read();
        std::string r;
        int pan;
        int tilt;
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
        return std::make_tuple(-1, -1);
    }

    /*
    Moves camera to home position.

        :return: True if successful, False if not.
        :rtype: bool
    */
    bool home() {
        ptContinuousMotion = false;
        return command("81010604FF");
    }

    /*
    Resets camera.

            :return: True if successful, False if not.
            :rtype: bool
    */
    bool reset() {
        ptContinuousMotion = false;
        zContinuous = false;
        return command("81010605FF");
    }

    /*
    Stops camera movement (pan/tilt).

        :return: True if successful, False if not.
        :rtype: bool
    */
    bool stop() {
        ptContinuousMotion = false;
        return command("8101060115150303FF");
    }

    /*
    Cancels current command.

        :return: True if successful, False if not.
        :rtype: bool
    */
    bool cancel() {
        ptContinuousMotion = false;
        zContinuous = false;
        return command("81010001FF");
    }

    /*
    Moves camera to absolute pan and tilt coordinates.

        :param speed: Speed (0-24)
        :param pan: numeric pan position
        :param tilt: numeric tilt position
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool goTo (const int pan, const int tilt, const size_t speed=5) {
        std::string comm = "81010602VVWWYYYYZZZZFF";
        formatCommand(comm, pan, tilt, speed);
        ptContinuousMotion = false;
        return command(comm);
    }

    /*
    Moves camera to relative pan and tilt coordinates.

        :param speed: Speed (0-24)
        :param pan: numeric pan adjustment
        :param tilt: numeric tilt adjustment
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool goToIncremental(const int pan, const int tilt, const size_t speed=5) {
        std::string comm = "81010603VVWWYYYYZZZZFF";
        formatCommand(comm, pan, tilt, speed);
        ptContinuousMotion = false;
        return command(comm);
    }

    /*
    Halt the zoom motor

        :return: True on success, False on failure
        :rtype: bool
    */
    bool zoomStop() {
        std::string comm = "8101040700FF";
        zContinuous = false;
        return command(comm);
    }

    /*
    Initiate tele zoom at speed range 0-7

        :param speed: zoom speed, 0-7
        :return: True on success, False on failure
        :rtype: bool
    */
    bool zoomIn(const size_t speed=0) {
        if (speed < 0 || speed > 7) { return false; }

        std::stringstream stream;
        stream << speed;
        std::string speed_hex = stream.str();
        std::string comm = "810104072pFF";
        comm.replace(9, 1, speed_hex);

        std::cout << "zoomIn comm string: " << comm << std::endl;
        zContinuous = true;
        return command(comm);
    }

    /*
    Moves camera to absolute zoom setting.

        :param zoom: numeric zoom position
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool zoomTo(const int zoom) {
        std::stringstream stream;
        stream << std::hex << zoom;
        std::string zoom_hex( stream.str() );
        if (zoom_hex.length() <= 3) {
            std::string zeros(4-zoom_hex.length(), '0');
            zoom_hex = zeros + zoom_hex;
        }
        stream.str(std::string());
        for (size_t index = 0; index < zoom_hex.length(); index++) {
            stream << "0" << zoom_hex[index];
        }
        zoom_hex = stream.str();

        std::string comm = "81010447pqrsFF";
        comm.replace(8, 4, zoom_hex);
        return command(comm);
    }

    /*
    Modifies pan speed to left.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool left(const size_t amount=5) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0103FF";
        comm.replace(8, 2, hex_string).replace(10, 2, "15");
        ptContinuousMotion = true;
        return command(comm);
    }

    /*
    Modifies pan speed to right

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool right(const size_t amount=5) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0203FF";
        comm.replace(8, 2, hex_string).replace(10, 2, "15");
        ptContinuousMotion = true;
        return command(comm);
    }

    /*
    Modifies pan speed to up.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool up(const size_t amount=5) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0301FF";
        comm.replace(8, 2, hex_string).replace(10, 2, "15");
        ptContinuousMotion = true;
        return command(comm);
    }

    /*
    Modifies pan speed to down.

        :param amount: Speed (0-24)
        :return: True if successful, False if not.
        :rtype: bool
    */
    bool down(const size_t amount=5) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0302FF";
        comm.replace(8, 2, hex_string).replace(10, 2, "15");
        ptContinuousMotion = true;
        return command(comm);
    }

    bool leftUp(const int pan, const int tilt) {
        std::string comm = "81010601VVWW0101FF";
        return move(comm, pan, tilt);
    }

    bool rightUp(const int pan, const int tilt) {
        std::string comm = "81010601VVWW0201FF";
        return move(comm, pan, tilt);
    }

    bool leftDown(const int pan, const int tilt) {
        std::string comm = "81010601VVWW0102FF";
        return move(comm, pan, tilt);
    }

    bool rightDown(const int pan, const int tilt) {
        std::string comm = "81010601VVWW0202FF";
        return move(comm, pan, tilt);
    }
};

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
