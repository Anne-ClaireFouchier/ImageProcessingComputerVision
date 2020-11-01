#include "ObjectTrackingKalmanFilter.hpp"

using namespace cv;


ObjectTrackingKalmanFilter::ObjectTrackingKalmanFilter (FilterType type) : type(type)
{
	if (type == FilterType::ConstantVelocity) {
		kalmanFilter.init (4, 2);
		kalmanFilter.transitionMatrix = (Mat_<float> (4, 4) << 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1);			// A
		kalmanFilter.processNoiseCov = (Mat_<float> (4, 4) << 25, 0, 0, 0, 0, 10, 0, 0, 0, 0, 25, 0, 0, 0, 0, 10);		// Q
		kalmanFilter.measurementMatrix = (Mat_<float> (2, 4) << 1, 0, 0, 0, 0, 0, 1, 0);								// H
	}
	else if (type == FilterType::ConstantAcceleration) {
		kalmanFilter.init (6, 2);
		kalmanFilter.transitionMatrix = (Mat_<float> (6, 6) << 1, 1, 0.5, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1);	
		kalmanFilter.processNoiseCov = (Mat_<float> (6, 6) << 25, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 25, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 1);
		kalmanFilter.measurementMatrix = (Mat_<float> (2, 6) << 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0);
	}
		
	setIdentity (kalmanFilter.errorCovPost, Scalar::all (1e5));							// P
	setIdentity (kalmanFilter.measurementNoiseCov, Scalar::all (25));					// R
}


// wrapper for prediction step of the kalman filter
void	ObjectTrackingKalmanFilter::PredictNextPosition ()
{
	Mat prediction = kalmanFilter.predict ();
	if (type == FilterType::ConstantVelocity)
		predictedTrajectory.push_back (Point2f (prediction.at<float> (0), prediction.at<float> (2)));
	else
		predictedTrajectory.push_back (Point2f (prediction.at<float> (0), prediction.at<float> (3)));

	finalTrajectory.push_back (predictedTrajectory.back ());
}


// wrapper for the correction step of the kalman filter
void	ObjectTrackingKalmanFilter::Correction (const cv::Point& measurement)
{
	measuredTrajectory.push_back (measurement);	

	Mat estimation = kalmanFilter.correct ((Mat_<float> (2, 1) << (float)measurement.x, (float)measurement.y));
	if (type == FilterType::ConstantVelocity)
		estimatedTrajectory.push_back (Point2f (estimation.at<float> (0), estimation.at<float> (2)));
	else
		estimatedTrajectory.push_back (Point2f (estimation.at<float> (0), estimation.at<float> (3)));

	// since we have an estimation we overwrite the prediction with the corrected version
	finalTrajectory.back () = estimatedTrajectory.back();
}


void	ObjectTrackingKalmanFilter::InitFilter (const Point& initialPosition)
{
	measuredTrajectory.push_back (initialPosition);

	setIdentity (kalmanFilter.errorCovPre);
	if (type == FilterType::ConstantVelocity) {
		kalmanFilter.statePost.at<float> (0) = initialPosition.x;
		kalmanFilter.statePost.at<float> (2) = initialPosition.y;
	}
	else {
		kalmanFilter.statePost.at<float> (0) = initialPosition.x;
		kalmanFilter.statePost.at<float> (3) = initialPosition.y;
	}
}


std::vector<Point>	ObjectTrackingKalmanFilter::GetMeasuredTrajectory () const
{
	return measuredTrajectory;
}


std::vector<Point>	ObjectTrackingKalmanFilter::GetPredictedTrajectory () const
{
	return predictedTrajectory;
}


std::vector<Point>	ObjectTrackingKalmanFilter::GetEstimatedTrajectory () const
{
	return estimatedTrajectory;
}

std::vector<Point>	ObjectTrackingKalmanFilter::GetFinalTrajectory () const
{
	return finalTrajectory;
}
