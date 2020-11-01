#ifndef OBJECTTRACKER_HPP
#define OBJECTTRACKER_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

#include "ShowManyImages.hpp"
#include "ObjectTrackingKalmanFilter.hpp"


typedef cv::Rect Blob;


class ObjectTracker {
public:
	ObjectTracker (std::string fileName, FilterType type);

	void		Process ();

private:
	void		ExtractForeground (const cv::Mat& frame, cv::Mat& fgmask, double learningRate);
	Blob		ExtractBiggestBlob (const cv::Mat& fgmask, cv::Size minSize) const;

	cv::Mat		PaintBlobOnImage (const cv::Mat& frame, const Blob& blob) const;
	cv::Mat		DrawTrajectories (const cv::Mat& frame) const;
	void		ShowFinalTrajectory (const cv::Mat& frame) const;
	void		ShowImages (const cv::Mat& fgmask,
							const cv::Mat& frame,
							const cv::Mat& frameWithBlob,
							const cv::Mat& trajectories) const;

private:
	cv::VideoCapture						cap;			// reader to grab videoframes
	cv::Ptr<cv::BackgroundSubtractorMOG2>	pMOG2;			// used for foreground sehmentation
	ObjectTrackingKalmanFilter				kalmanFilter;	// costumized kalman filter

	std::string								title = "Kalman filtering for object tracking";
	std::string								finalTitle = "Final Trajectory";
};


#endif // OBJECTTRACKER_HPP

