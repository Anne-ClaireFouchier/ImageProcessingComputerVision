#include "FusionModel.hpp"

using namespace cv;

// Initialize the two models
FusionModel::FusionModel (const Mat& frame, const Rect& initialPos, int nrBinsColor, int nrBinsGradient, char histType) : 
	Model(initialPos, nrBins),
	colorModel(frame, initialPos, nrBinsColor, histType),
	gradientModel(frame, initialPos, nrBinsGradient)
{
}

// forward call to the stored model
double	FusionModel::CalculateColorDifference (const Mat& patch) const
{
	return colorModel.CalculateColorDifference (patch);
}

// forward call to the stored model
double	FusionModel::CalculateGradientDifference (const Mat& patch) const
{
	return gradientModel.CalculateGradientDifference (patch);
}

Mat		FusionModel::CalculateFeature (const Mat& patch) const
{
	// no separate histogram needed for this model, they are contained in the respective models
	return Mat ();
}

Mat FusionModel::ConvertPatch (const Mat& patch) const
{
	// no separate histogram needed for this model, they are contained in the respective models
	return Mat ();
}
