#pragma once

#include <opencv2/opencv.hpp>
#include "utils.hpp"

class Model {
public:
	Model (const cv::Rect& initialPos, int nrBins);
	virtual ~Model();
	
	virtual double		CalculateColorDifference (const cv::Mat& patch) const;
	virtual double		CalculateGradientDifference (const cv::Mat& patch) const;
	virtual void		UpdateModel (const cv::Rect& loc, const cv::Mat& patch);
	
	cv::Rect			GetLocation () const;
	cv::Mat				GetFeature () const;

//protected:
	virtual cv::Mat		CalculateFeature (const cv::Mat& patch) const = 0;
	virtual cv::Mat		ConvertPatch (const cv::Mat& patch) const = 0;

protected:
	cv::Rect			location;
	cv::Mat				feature;
	int					nrBins;

};

