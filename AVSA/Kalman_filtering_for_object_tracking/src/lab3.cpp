//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>

// additional libraries
#include "ObjectTracker.hpp"


// main function
int main (int argc, char** argv)
{
	if (argc != 3) {
		std::cerr << "wrong number of arguments\nYou need two arguments: location of the video and the filtertype (ConstantVelocity or ConstantAcceleration)";
		return -1;
	}

	if (argc == 3 && argv[1] == "help") {
		std::cout << "You need two arguments: location of the video and the filtertype (ConstantVelocity or ConstantAcceleration)";
		return 0;
	}

	FilterType type;
	if (string(argv[2]) == "ConstantVelocity")
		type = FilterType::ConstantVelocity;
	else if (string(argv[2]) == "ConstantAcceleration")
		type = FilterType::ConstantAcceleration;
	else {
		std::cerr << "Wrong type of filter";
		return -1;
	}

	ObjectTracker tracker (argv[1], type);
	tracker.Process ();

	return 0;
}




