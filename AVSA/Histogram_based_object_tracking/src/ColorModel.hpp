#pragma once

#include "Model.hpp"

class ColorModel : public Model {
public:
	ColorModel (const cv::Mat& frame, const cv::Rect& initialPos, int nrBins, char histType);

	virtual double		CalculateColorDifference (const cv::Mat& patch) const;

//protected:
	virtual cv::Mat		CalculateFeature (const cv::Mat& patch) const;
	virtual cv::Mat		ConvertPatch (const cv::Mat& patch) const;

private:
	char				histType;
};

