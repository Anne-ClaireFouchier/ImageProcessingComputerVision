#include "ColorModel.hpp"

using namespace cv;
using namespace std;

// initialize the model and parameters
ColorModel::ColorModel (const Mat& frame, const Rect& initialPos, int nrBins, char histType) : Model (initialPos, nrBins), histType(histType)
{
	feature = CalculateFeature (frame (initialPos));
}

// calculate the difference between two histograms using the Bhattacharyya difference
double	ColorModel::CalculateColorDifference (const Mat& patch) const
{
	Mat histogram2 = CalculateFeature (patch);
	return compareHist (feature, histogram2, HISTCMP_BHATTACHARYYA);
}

// calculate the histogram according to the channel used
Mat		ColorModel::CalculateFeature (const Mat& patch) const
{
	Mat hist = Mat::zeros (Size (nrBins, 1), CV_64FC1);;
	Mat convertedPatch = ConvertPatch(patch);
	switch (histType) {
		case 'K':
		case 'B':
		case 'G':
		case 'R':
		case 'S':
			hist = CalculateHistogram (convertedPatch, 256, nrBins);
			break;
		case 'H':
			hist = CalculateHistogram (convertedPatch, 180, nrBins);
			break;
		default:
			cerr << "Invalid hist type!";
			break;
	}

	normalize (hist, hist, 1, 0, NORM_L1, -1, Mat ());

	return hist;
}

// convert the given patch to the desired channel
Mat ColorModel::ConvertPatch (const Mat& patch) const
{
	Mat convertedPatch;
	vector<Mat> channels;
	switch (histType) {
		case 'K':
			cvtColor (patch, convertedPatch, CV_BGR2GRAY);
			break;
		case 'B':
			split (patch, channels);
			convertedPatch = channels[0];
			break;
		case 'G':
			split (patch, channels);
			convertedPatch = channels[1];
			break;
		case 'R':
			split (patch, channels);
			convertedPatch = channels[2];
			break;
		case 'H':
			cvtColor (patch, convertedPatch, CV_BGR2HSV);
			split (convertedPatch, channels);
			convertedPatch = channels[0];
			break;
		case 'S':
			cvtColor (patch, convertedPatch, CV_BGR2HSV);
			split (convertedPatch, channels);
			convertedPatch = channels[1];
			break;
		default:
			cerr << "Invalid hist type!";
			break;
	}
	return convertedPatch;
}
