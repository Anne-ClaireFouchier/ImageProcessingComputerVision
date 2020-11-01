#include <opencv2/opencv.hpp>

#ifndef GMM_H_INCLUDE
#define GMM_H_INCLUDE

//#pragma once

// class representing a Gaussian distributin
class Gaussian {
public:
	double mean;
	double std;
	double weight;

	// random initialization
	Gaussian ();
	// initialize with concrete values
	Gaussian (double mean, double std, double weight);

	// check if a given pixel matches the Gaussian
	bool match (double value) const;

	// used to order the Gaussians according to their weight
	bool operator < (const Gaussian& rhs) const;
};

// per pixel, it contains the K Gaussians
class gmm {
public:
	// stores the K nr of Gaussians
	std::vector<Gaussian> gaussians;

	// initializes the Gaussians
	gmm (uchar value, int K);

	// background pixel update
	void update (double value);
	// checks if pixel mathces a background gaussian
	bool isItBackground (double value, double Wth);
	// foreground pixel update, repalces the lowest weight gaussian
	void addGaussian (double value);
	// checks if any of the gaussians matches the pixel
	bool hasMatchingGaussian (double value);

	//normalizes the weights to 1
	void normalizeWeights ();
};

// per image, represents the baclground model
class GMM {
public:
	// learning rate
	static double alpha;

	// model representation
	std::vector<std::vector<gmm>> model;
	// threshold determining the baclground gaussians
	double Wth;

	// almost empty constructor, init needed before use
	GMM ();
	// initializes the background model and sets the parameters
	void init (const cv::Mat& img, double Wth, int K);

	// generates the froeground mask
	void getFGMask (const cv::Mat& frame, cv::Mat& fgmask);
};


#endif
