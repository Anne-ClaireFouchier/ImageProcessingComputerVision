#pragma once

#include "Model.hpp"
#include "ColorModel.hpp"
#include "GradientModel.hpp"

class FusionModel : public Model {
public:
	FusionModel (const cv::Mat& frame, const cv::Rect& initialPos, int nrBinsColor, int nrBinsGradient, char histType);

	virtual double		CalculateColorDifference (const cv::Mat& patch) const;
	virtual double		CalculateGradientDifference (const cv::Mat& patch) const;

//protected:
	virtual cv::Mat		CalculateFeature (const cv::Mat& patch) const;
	virtual cv::Mat		ConvertPatch (const cv::Mat& patch) const;

private:
	ColorModel			colorModel;
	GradientModel		gradientModel;
};

