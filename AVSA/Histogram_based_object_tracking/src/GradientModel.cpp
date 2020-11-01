#include "GradientModel.hpp"

using namespace cv;

// initialize parameters, model and the HOGDescriptor
GradientModel::GradientModel (const Mat& frame, const Rect& initialPos, int nrBins) : Model (initialPos, nrBins)
{
	hog = HOGDescriptor ();
	hog.nbins = nrBins;
	feature = CalculateFeature (frame (initialPos));
}

// implement an L2 difference between the descriptors
double	GradientModel::CalculateGradientDifference (const Mat& patch) const
{
	Mat feature2 = CalculateFeature (patch);
	return norm (feature, feature2, NORM_L2);
}

// calculate the descriptors for a given patch
Mat		GradientModel::CalculateFeature (const Mat& patch) const
{
	std::vector<float> descriptors;
	Mat resizedPatch = ConvertPatch(patch);
	hog.compute (resizedPatch, descriptors);
	return Mat(descriptors).clone();
}

// resize the patch to fit the hogdescriptor computation
Mat		GradientModel::ConvertPatch (const Mat& patch) const
{
	Mat resizedPatch;
	resize (patch, resizedPatch, Size (64, 128));
	return resizedPatch;
}
