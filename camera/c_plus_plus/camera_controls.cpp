#include "camera_controls.h"

namespace Capstone {

    UDPCamera::UDPCamera (const std::string& host, const std::string& port)
        : udp_host(host), udp_port(port), io(), socket(io) { }

    void UDPCamera::connect() {
        try {
            std::cout << "Connecting to camera..." << std::endl;
            boost::asio::ip::udp::resolver resolver(io);
    		boost::asio::ip::udp::resolver::query query(boost::asio::ip::udp::v4(), udp_host, udp_port);
    		receiver_endpoint = *resolver.resolve(query);
    		socket.open(boost::asio::ip::udp::v4());
            std::cout << "Connected to camera successfully" << std::endl;
        } catch (std::exception& e){
            std::cerr << e.what() << std::endl;
        }
    }

    bool UDPCamera::command(const std::string& commStr) {
        std::vector<uint8_t> comm;
        boost::algorithm::unhex(commStr.begin(), commStr.end(), std::back_inserter(comm));
        // std::cout << "-----------------Sending the following -----------------" << '\n';
        // std::cout << commStr << '\n';
        // std::cout << std::endl;
        // std::cout << "=====================Done===========================" << '\n';
        try {
    		socket.send_to(boost::asio::buffer(comm), receiver_endpoint);
            return true;
        } catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
            return false;
        }
    }

    std::string UDPCamera::read() {
        std::string res;
        socket.receive_from(
            boost::asio::buffer(recv_buffer),
            receiver_endpoint);
        boost::algorithm::hex(recv_buffer.begin(), recv_buffer.end(), back_inserter(res));
        return res;
    }

    void UDPCamera::end() {
        boost::system::error_code ec;
        socket.shutdown(boost::asio::ip::udp::socket::shutdown_both, ec);
        socket.close();
    }

    bool PTZOptics20x::move(std::string& comm, const size_t pan, const size_t tilt) {
        //
        // std::cout << "pan: " << pan << '\n';
        // std::cout << "tilt: " << tilt << '\n';

        std::stringstream stream;
        stream << std::hex << pan;
        std::string h1( stream.str() );

        stream.str(std::string());
        stream << std::hex << tilt;
        std::string h2( stream.str() );

        if (h1.length() < 2) { h1.insert(0, "0"); }
        if (h2.length() < 2) { h2.insert(0, "0"); }
        ptContinuousMotion = true;
        // std::cout << comm << '\n';
        comm.replace(8, 2, h1).replace(10, 2, h2);
        // std::cout << comm << '\n';
        return command(comm);
    }

    void PTZOptics20x::formatCommand(std::string& comm, const int pan,const int tilt, const size_t speed) {
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

    PTZOptics20x::PTZOptics20x(const std::string& host, const std::string& port)
        : UDPCamera(host, port) {}

    bool PTZOptics20x::panTiltOngoing() {
        return ptContinuousMotion;
    }

    bool PTZOptics20x::zoomOnGoing() {
        return zContinuous;
    }

    std::tuple<int, int> PTZOptics20x::getPanTiltPosition() {
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

    bool PTZOptics20x::home() {
        ptContinuousMotion = false;
        return command("81010604FF");
    }


    bool PTZOptics20x::reset() {
        ptContinuousMotion = false;
        zContinuous = false;
        return command("81010605FF");
    }

    bool PTZOptics20x::stop() {
        ptContinuousMotion = false;
        return command("8101060115150303FF");
    }

    bool PTZOptics20x::cancel() {
        ptContinuousMotion = false;
        zContinuous = false;
        return command("81010001FF");
    }

    bool PTZOptics20x::goTo (const int pan, const int tilt, const size_t speed) {
        std::string comm = "81010602VVWWYYYYZZZZFF";
        formatCommand(comm, pan, tilt, speed);
        ptContinuousMotion = false;
        return command(comm);
    }

    bool PTZOptics20x::goToIncremental(const int pan, const int tilt, const size_t speed) {
        std::string comm = "81010603VVWWYYYYZZZZFF";
        formatCommand(comm, pan, tilt, speed);
        ptContinuousMotion = false;
        return command(comm);
    }

    bool PTZOptics20x::zoomStop() {
        std::string comm = "8101040700FF";
        zContinuous = false;
        return command(comm);
    }

    bool PTZOptics20x::zoomIn(const size_t speed) {
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

    bool PTZOptics20x::zoomTo(const int zoom) {
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

    bool PTZOptics20x::left(const size_t amount) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0103FF";
        comm.replace(8, 2, hex_string).replace(10, 2, "10");
        ptContinuousMotion = true;
        return command(comm);
    }

    bool PTZOptics20x::right(const size_t amount) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0203FF";
        comm.replace(8, 2, hex_string).replace(10, 2, "10");
        ptContinuousMotion = true;
        return command(comm);
    }

    bool PTZOptics20x::up(const size_t amount) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0301FF";
        comm.replace(8, 2, "10").replace(10, 2, hex_string);
        ptContinuousMotion = true;
        return command(comm);
    }

    bool PTZOptics20x::down(const size_t amount) {
        std::stringstream stream;
        stream << std::hex << amount;
        std::string hex_string = stream.str();
        if (hex_string.length() < 2) {
            hex_string = "0" + hex_string;
        }
        std::string comm = "81010601VVWW0302FF";
        comm.replace(8, 2, "10").replace(10, 2, hex_string);
        ptContinuousMotion = true;
        return command(comm);
    }

    bool PTZOptics20x::leftUp(const size_t pan, const size_t tilt) {
        std::string comm = "81010601VVWW0101FF";
        return move(comm, pan, tilt);
    }

    bool PTZOptics20x::rightUp(const size_t pan, const size_t tilt) {
        std::string comm = "81010601VVWW0201FF";
        return move(comm, pan, tilt);
    }

    bool PTZOptics20x::leftDown(const size_t pan, const size_t tilt) {
        std::string comm = "81010601VVWW0102FF";
        return move(comm, pan, tilt);
    }

    bool PTZOptics20x::rightDown(const size_t pan, const size_t tilt) {
        std::string comm = "81010601VVWW0202FF";
        return move(comm, pan, tilt);
    }
}
