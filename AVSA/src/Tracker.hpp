#pragma once

#include "Model.hpp"
#include <opencv2/opencv.hpp>


class Tracker {
public:
	Tracker (Model* model, unsigned int nrCandidates, unsigned int stride);
	virtual ~Tracker ();

	cv::Rect		GetEstimate (const cv::Mat& frame) const;

protected:
	Model*			model;
	unsigned int	nrCandidates;
	unsigned int	stride;

};

