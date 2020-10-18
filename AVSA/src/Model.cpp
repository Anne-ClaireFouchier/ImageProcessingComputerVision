#include "Model.hpp"

Model::Model (const cv::Rect& initialPos, int nrBins)
{
	location = initialPos;
	this->nrBins = nrBins;
}

Model::~Model() 
{
}

double Model::CalculateColorDifference (const cv::Mat& patch) const
{
	return 0.0;
}

double Model::CalculateGradientDifference (const cv::Mat& patch) const
{
	return 0.0;
}

void Model::UpdateModel (const cv::Rect& loc, const cv::Mat& patch)
{
	location = loc;
	//feature = CalculateFeature (patch);
}

cv::Rect	Model::GetLocation () const
{
	return location;
}

cv::Mat Model::GetFeature () const
{
	return feature;
}

