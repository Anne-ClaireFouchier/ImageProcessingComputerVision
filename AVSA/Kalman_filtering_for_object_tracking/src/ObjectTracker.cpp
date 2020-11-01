#include "ObjectTracker.hpp"


using namespace cv;
using namespace std;


ObjectTracker::ObjectTracker (string fileName, FilterType type) : kalmanFilter(type)
{
	// try opening the video file
	cap.open (fileName);
	if (!cap.isOpened ())
		cerr << "Could not open video file " << fileName << endl;

	// inititalize the background subtractor
	pMOG2 = createBackgroundSubtractorMOG2 ();
	pMOG2 = createBackgroundSubtractorMOG2 ();
	pMOG2->setVarThreshold (6);
	pMOG2->setHistory (50);
}


void ObjectTracker::Process ()
{
	// if opening the video failed we don't do anything
	if (!cap.isOpened ())
		return;

	Mat frame;		// current Frame
	Mat fgmask;		// foreground mask

	Mat img;		// current Frame
	Blob object;	// tracked object
	bool found = false;

	while (true) {
		// get frame
		cap >> img;

		// check if we achieved the end of the file (e.g. img.data is empty)
		if (!img.data)
			break;

		img.copyTo (frame);

		// until there hasn't been a ball in the scene we don't start the kalman filter predictions
		if (found)
			kalmanFilter.PredictNextPosition ();

		// extract the foreground and the biggest blob from it
		ExtractForeground (frame, fgmask, 0.001);
		object = ExtractBiggestBlob (fgmask, Size(50, 50));

		// check if an object is detected, meaning the biggest blob (above the min size) is not a zero size blob
		if (object.area () > 0) {
			Point measurement = Point (object.x + object.width / 2, object.y + object.height / 2);

			// this is the first detection of an object, haven't found  one before
			// so the filter is initialized with this first position
			if (!found) {
				kalmanFilter.InitFilter (measurement);
				found = true;
			}
			else {
				// since we have a measurement we can apply correction
				kalmanFilter.Correction (measurement);
			}
		}

		// show the current measurements
		//ShowImages (fgmask, frame, PaintBlobOnImage (frame, object), Mat());
		ShowImages (fgmask, frame, PaintBlobOnImage (frame, object), DrawTrajectories (frame));

		// exit if ESC key is pressed
		if (waitKey (30) == 27)
			break;
	} // main loop

	// show trajectories
	ShowImages (fgmask, frame, PaintBlobOnImage (frame, object), DrawTrajectories (frame));
	ShowFinalTrajectory (frame);

	waitKey (0);
	// release all resources
	cap.release ();
	destroyAllWindows ();
}


// applies the background subtractor and a morphological opening to get the foreground
void	ObjectTracker::ExtractForeground (const Mat& frame, Mat& fgmask, double learningRate)
{
	pMOG2->apply (frame, fgmask, learningRate);
	int morph_size = 3;
	Mat kernel = getStructuringElement (MORPH_RECT, Size (2 * morph_size + 1, 2 * morph_size + 1), Point (morph_size, morph_size));
	morphologyEx (fgmask, fgmask, MORPH_OPEN, kernel);
}


Blob	ObjectTracker::ExtractBiggestBlob (const Mat& fgmask, Size minSize) const
{
	Mat aux; // image to be updated each time a blob is detected (blob cleared)
	fgmask.convertTo (aux, CV_32SC1);

	//Connected component analysis
	Blob rect;
	Blob result;

	// traverse the image looking for foreground pixels
	for (int i = 0; i < aux.rows; i++) {
		for (int j = 0; j < aux.cols; j++) {
			// if a foreground pixel is found we call one of the fill methods to fill the blob with the given value and determine
			// the corresponding bounding box
			if (aux.at<int> (i, j) == 255) {
				floodFill (aux, Point (j, i), 1024, &rect, Scalar (), Scalar (), 8);
				
				// if a blob bigger than the current biggest is found (and is bigger than min size) than it's tored instead
				if (rect.size ().area () > result.area () && rect.size ().width > minSize.width && rect.size().height > minSize.height)
					result = rect;
			}
		}
	}

	return result;
}


// paint a given blob on the frame
Mat		ObjectTracker::PaintBlobOnImage (const Mat& frame, const Blob& blob) const
{
	Mat result;
	frame.copyTo (result);
	circle (result, Point (blob.x + blob.width / 2, blob.y + blob.height / 2), blob.width / 2, Scalar (0, 0, 255), 3);
	return result;
}


// draws all the saved trajectories on the frame
Mat		ObjectTracker::DrawTrajectories (const Mat& frame) const
{
	Mat result;
	int radius = 10;
	Scalar red = Scalar (0, 0, 255);
	Scalar green = Scalar (0, 255, 0);
	Scalar blue = Scalar (255, 0, 0);
	frame.copyTo (result);	

	for (Point point : kalmanFilter.GetPredictedTrajectory()) {
		circle (result, point, radius, green, 1);
		line (result, Point (point.x - radius / 2, point.y), Point (point.x + radius / 2, point.y), green, 1);
		line (result, Point (point.x, point.y - radius / 2), Point (point.x, point.y + radius / 2), green, 1);
	}

	for (Point point : kalmanFilter.GetEstimatedTrajectory()) {
		circle (result, point, radius, blue, 2);
		line (result, Point (point.x - radius / 2, point.y), Point (point.x + radius / 2, point.y), blue, 1);
		line (result, Point (point.x, point.y - radius / 2), Point (point.x, point.y + radius / 2), blue, 1);
	}

	for (Point point : kalmanFilter.GetMeasuredTrajectory()) {
		circle (result, point, 5, red, 2);
		line (result, Point (point.x - radius / 2, point.y), Point (point.x + radius / 2, point.y), red, 1);
		line (result, Point (point.x, point.y - radius / 2), Point (point.x, point.y + radius / 2), red, 1);
	}

	return result;
}


// shows the final trajectory in a new window
void	ObjectTracker::ShowFinalTrajectory (const Mat& frame) const
{
	Mat result;
	int radius = 5;
	Scalar red = Scalar (0, 0, 255);
	Scalar blue = Scalar (255, 0, 0);
	frame.copyTo (result);

	vector<Point> measurements = kalmanFilter.GetMeasuredTrajectory ();
	vector<Point> predictions = kalmanFilter.GetFinalTrajectory ();

	for (Point point : measurements) {
		line (result, Point (point.x - radius / 2, point.y), Point (point.x + radius / 2, point.y), red, 1);
		line (result, Point (point.x, point.y - radius / 2), Point (point.x, point.y + radius / 2), red, 1);
	}

	for (Point point : predictions) {
		circle (result, point, 2, blue, 0.5);
	}

	for (size_t i = 0; i < measurements.size () - 1; i++) {
		line (result, measurements[i], measurements[i + 1], red, 1);
	}

	for (size_t i = 0; i < predictions.size () - 1; i++) {
		line (result, predictions[i], predictions[i + 1], blue, 1);
	}

	putText (result, "Measurements", Point (5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar (0, 0, 255));
	putText (result, "Final Predictions", Point (5, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar (255, 0, 0));

	namedWindow (finalTitle, WINDOW_AUTOSIZE);
	imshow (finalTitle, result);
}


// wrapper for ShowManyImages which puts the texts for the images
void ObjectTracker::ShowImages (const Mat& fgmask, const Mat& frame, const Mat& frameWithBlob, const Mat& trajectories) const
{
	Mat fgmaskWithText; fgmask.copyTo (fgmaskWithText);
	Mat frameWithText; frame.copyTo (frameWithText);
	Mat frameWithBlobWithText; frameWithBlob.copyTo (frameWithBlobWithText);
	Mat trajectoriesWithText; trajectories.copyTo (trajectoriesWithText);

	if (trajectories.empty ()) {
		trajectoriesWithText = Mat::zeros (frame.size (), CV_8UC1);
	}
	else {
		putText (trajectoriesWithText, "Measurements", Point (5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar (0, 0, 255));
		putText (trajectoriesWithText, "Predictions", Point (5, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar (0, 255, 0));
		putText (trajectoriesWithText, "Estimations", Point (5, 45), FONT_HERSHEY_SIMPLEX, 0.5, Scalar (255, 0, 0));
	}

	putText (fgmaskWithText, "FG Mask", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
	putText (frameWithText, "Current Frame", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
	putText (frameWithBlobWithText, "Current Frame with Detected Object", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

	ShowManyImages (title, 4, fgmaskWithText, frameWithText, frameWithBlobWithText, trajectoriesWithText);
}
