#include "gmm.hpp"
#include <random>

Gaussian::Gaussian ()
{
	mean = rand () % 256;
	std = 30.0; // for the random init a bit higher deviation
	weight = 0.1; // and lower weight
}

Gaussian::Gaussian (double mean, double std, double weight) : mean(mean), std(std), weight(weight)
{
}

bool Gaussian::match (double value) const
{
	return abs (mean - value) < 3 * std; // checking if mathces the gaussian
}

bool Gaussian::operator<(const Gaussian& rhs) const
{
	return weight < rhs.weight;
}

gmm::gmm (uchar value, int K)
{
	// the first gaussian is initialized with the actual mean, a lower deviation and higher weight
	gaussians.push_back (Gaussian (value, 20, 1));
	// the rest are initialized randomly
	for (size_t i = 1; i < K; i++)
		gaussians.push_back (Gaussian ());

	normalizeWeights ();
}

void gmm::update (double value)
{
	for (Gaussian& g : gaussians) {
		// only update the matching gaussians according to the formula
		if (g.match (value)) {
			g.mean = GMM::alpha * value + (1.0 - GMM::alpha) * g.mean;
			g.std = sqrt (GMM::alpha * pow (value - g.mean, 2) + (1.0 - GMM::alpha) * pow (g.std, 2));
		}
		else {
			// the non-matching ones get lower weight
			g.weight *= (1.0 - GMM::alpha);
		}
	}

	// the matching ones' weight was not increased but the normalization will actually do that
	normalizeWeights ();
	//reorder according to weights
	std::sort (gaussians.begin (), gaussians.end ());
	std::reverse (gaussians.begin (), gaussians.end ());
}

bool gmm::isItBackground (double value, double Wth)
{
	double weights = 0.0;

	// check if matches a gaussian from the ones that add upp to Wth
	for (size_t i = 0; weights < Wth && i < gaussians.size (); i++) {
		if (gaussians[i].match (value))
			return true;

		weights += gaussians[i].weight;
	}

	return false;
}

void gmm::addGaussian (double value)
{
	// overwrite lowest weight gaussian, the weight stays the same
	gaussians[gaussians.size () - 1].mean = value;
	gaussians[gaussians.size () - 1].std = 30;
}

bool gmm::hasMatchingGaussian (double value)
{
	for (Gaussian& g : gaussians)
		if (g.match (value))
			return true;

	return false;
}

void gmm::normalizeWeights ()
{
	double sum = 0;
	for (Gaussian& g : gaussians)
		sum += g.weight;

	for (Gaussian& g : gaussians)
		g.weight /= sum;
}


double GMM::alpha = 0.01;

GMM::GMM ()
{
	Wth = 1;
}

void GMM::init (const cv::Mat& img, double Wth, int K)
{
	this->Wth = Wth;
	// initialize each pixel of the model with the K gaussians
	for (int i = 0; i < img.rows; i++) {
		std::vector<gmm> row;
		for (int j = 0; j < img.cols; j++) {
			row.push_back (gmm (img.at<uchar> (i, j), K));
		}
		model.push_back (row);
	}
}

void GMM::getFGMask (const cv::Mat& frame, cv::Mat& fgmask)
{
	fgmask = cv::Mat::zeros (frame.size (), CV_8UC1);

	for (int i = 0; i < frame.rows; i++) {
		for (int j = 0; j < frame.cols; j++) {
			// if the pixel is background it is set in the fgmask and a background update is performed on the gaussians
			if (model[i][j].isItBackground (frame.at<uchar> (i, j), Wth)) {
				fgmask.at<uchar> (i, j) = 0;
				model[i][j].update (frame.at<uchar> (i, j));
			}
			else {
				// if it's a foreground it is set in fgmask
				fgmask.at<uchar> (i, j) = 255;
				// a blind update is performed, if it has matching gaussians it is updated as a background pixel 
				if (model[i][j].hasMatchingGaussian (frame.at<uchar> (i, j)))
					model[i][j].update (frame.at<uchar> (i, j));
				else // otherwise as a foreground pixel
					model[i][j].addGaussian (frame.at<uchar> (i, j));
			}
		}
	}
}
