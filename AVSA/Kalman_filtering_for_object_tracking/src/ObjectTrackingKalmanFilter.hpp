#ifndef OBJECTTRACKINGKALMANFILTER_H_INCLUDE
#define OBJECTTRACKINGKALMANFILTER_H_INCLUDE

#include "opencv2/opencv.hpp"


enum class FilterType {
	ConstantVelocity = 0,
	ConstantAcceleration = 1
};


class ObjectTrackingKalmanFilter {
public:
	ObjectTrackingKalmanFilter (FilterType type);

	void						PredictNextPosition ();
	void						Correction (const cv::Point& measurement);
	void						InitFilter (const cv::Point& initialPosition);

	std::vector<cv::Point>		GetMeasuredTrajectory () const;
	std::vector<cv::Point>		GetPredictedTrajectory () const;
	std::vector<cv::Point>		GetEstimatedTrajectory () const;
	std::vector<cv::Point>		GetFinalTrajectory () const;

private:
	cv::KalmanFilter			kalmanFilter;
	FilterType					type;

	std::vector<cv::Point>		measuredTrajectory;
	std::vector<cv::Point>		predictedTrajectory;
	std::vector<cv::Point>		estimatedTrajectory;
	std::vector<cv::Point>		finalTrajectory;

	size_t						currentStep = 0;
};

#endif

