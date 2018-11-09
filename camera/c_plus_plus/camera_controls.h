#ifndef CAMERA_CONTROLS_H
#define CAMERA_CONTROLS_H

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

namespace Capstone {
    class UDPCamera {
    private:
        const std::string udp_host;
        const std::string udp_port;
        boost::asio::io_service io;
        boost::asio::ip::udp::socket socket;
        boost::asio::ip::udp::endpoint receiver_endpoint;
        boost::array<char, 16> recv_buffer;

    public:
        /*
        PTZOptics VISCA control class.

            :param host: UDP control host.
            :type host: str
            :param port: UDP control port.
            :type port: str
        */
        UDPCamera (const std::string& host, const std::string& port);

        /*
        Initializes camera object by establishing TCP control session.

            :return: Camera object.
            :rtype: TCPCamera
        */
        void connect();

        /*
        Sends hexadecimal string to TCP control socket.

            :param com: Command string. Hexadecimal format.
            :type com: str
            :return: Success.
            :rtype: bool
        */
        bool command(const std::string& commStr);

        std::string read();

        void end();

    };

    class PTZOptics20x : public UDPCamera {
    private:
        bool ptContinuousMotion = false;
        bool zContinuous = false;

        bool move(std::string& comm, const int pan, const int tilt);

        void formatCommand(std::string& comm, const int pan,const int tilt, const size_t speed);

    public:
        PTZOptics20x(const std::string& host, const std::string& port);

        bool panTiltOngoing();

        bool zoomOnGoing();

        /*
        Retrieves current pan/tilt position.
        Pan is 0 at home. Right is positive, max 2448. Left ranges from full left 63088 to 65555 before home.
        Tilt is 0 at home. Up is positive, max 1296. Down ranges from fully depressed at 65104 to 65555 before home.

            :return: pan position
            :rtype: int
            :return: tilt position
            :rtype: int
        */
        std::tuple<int, int> getPanTiltPosition();

        /*
        Moves camera to home position.

            :return: True if successful, False if not.
            :rtype: bool
        */
        bool home();

        /*
        Resets camera.

                :return: True if successful, False if not.
                :rtype: bool
        */
        bool reset();

        /*
        Stops camera movement (pan/tilt).

            :return: True if successful, False if not.
            :rtype: bool
        */
        bool stop();

        /*
        Cancels current command.

            :return: True if successful, False if not.
            :rtype: bool
        */
        bool cancel();

        /*
        Moves camera to absolute pan and tilt coordinates.

            :param speed: Speed (0-24)
            :param pan: numeric pan position
            :param tilt: numeric tilt position
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool goTo (const int pan, const int tilt, const size_t speed=5);

        /*
        Moves camera to relative pan and tilt coordinates.

            :param speed: Speed (0-24)
            :param pan: numeric pan adjustment
            :param tilt: numeric tilt adjustment
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool goToIncremental(const int pan, const int tilt, const size_t speed=5);

        /*
        Halt the zoom motor

            :return: True on success, False on failure
            :rtype: bool
        */
        bool zoomStop();

        /*
        Initiate tele zoom at speed range 0-7

            :param speed: zoom speed, 0-7
            :return: True on success, False on failure
            :rtype: bool
        */
        bool zoomIn(const size_t speed=0);

        /*
        Moves camera to absolute zoom setting.

            :param zoom: numeric zoom position
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool zoomTo(const int zoom);

        /*
        Modifies pan speed to left.

            :param amount: Speed (0-24)
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool left(const size_t amount=5);

        /*
        Modifies pan speed to right

            :param amount: Speed (0-24)
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool right(const size_t amount=5);

        /*
        Modifies pan speed to up.

            :param amount: Speed (0-24)
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool up(const size_t amount=5);

        /*
        Modifies pan speed to down.

            :param amount: Speed (0-24)
            :return: True if successful, False if not.
            :rtype: bool
        */
        bool down(const size_t amount=5);

        bool leftUp(const int pan, const int tilt);

        bool rightUp(const int pan, const int tilt);

        bool leftDown(const int pan, const int tilt);

        bool rightDown(const int pan, const int tilt);
    };
}

#endif
