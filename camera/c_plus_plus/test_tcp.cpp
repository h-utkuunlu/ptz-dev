#include <iostream>
#include <exception>
#include <array>
#include <string>
#include <boost/asio.hpp>

namespace ba = boost::asio;
using ba::ip::tcp;
using namespace std;

void client(ba::io_service& io, const std::string& host, const std::string& port) {
	try {
		tcp::resolver resolver(io);
		tcp::resolver::query query(host, port);
		tcp::socket socket(io);
		ba::connect(socket, resolver.resolve(query));

		uint8_t foo[] = {0x81, 0x01, 0x06, 0x04, 0xFF};
		std::cout << sizeof(foo[0]);
		const size_t bytes = boost::asio::write(socket, boost::asio::buffer( foo ));
    	std::cout << "sent " << bytes << " bytes" << std::endl;
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
}

int main(int argc, char* argv[])
{
	ba::io_service io;
    const std::string PORT("5678");
    const std::string HOSTNAME("192.168.1.40");
	client(io, HOSTNAME, PORT);

}
