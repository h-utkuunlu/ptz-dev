#include <iostream>
#include <exception>
#include <array>
#include <string>
#include <boost/asio.hpp>

namespace ba = boost::asio;
using ba::ip::udp;
using namespace std;

void client(ba::io_service& io, const std::string& host, const std::string& port) {
	try {
		udp::resolver resolver(io);
		udp::resolver::query query(udp::v4(), host, port);
		udp::endpoint receiver_endpoint = *resolver.resolve(query);
		udp::socket socket(io);
		socket.open(udp::v4());

		uint8_t foo[] = {0x81, 0x01, 0x06, 0x04, 0xFF};
		socket.send_to(boost::asio::buffer(foo), receiver_endpoint);
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
}

int main(int argc, char* argv[])
{
	ba::io_service io;
    const std::string PORT("1259");
    const std::string HOSTNAME("192.168.1.40");
	client(io, HOSTNAME, PORT);

}
