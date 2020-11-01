#pragma once

#include "Model.hpp"
//#include <opencv2/opencv.hpp>

class GradientModel : public Model {
public:
	GradientModel (const cv::Mat& frame, const cv::Rect& initialPos, int nrBins);

	virtual double		CalculateGradientDifference (const cv::Mat& patch) const;

//protected:
	virtual cv::Mat		CalculateFeature (const cv::Mat& patch) const;
	virtual cv::Mat		ConvertPatch (const cv::Mat& patch) const;

private:
	cv::HOGDescriptor	hog;
};

