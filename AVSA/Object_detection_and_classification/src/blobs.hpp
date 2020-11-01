#ifndef BLOBS_H_INCLUDE
#define BLOBS_H_INCLUDE

#include "opencv2/opencv.hpp"


using namespace cv;


enum class FillModes {
	opencvFill		= 0,
	recursiveFill	= 1,
	sequentialFill	= 2
};


/// Type of labels for blobs
enum class Class {
	UNKNOWN		= 0,
	PERSON		= 1,
	CAR			= 2,
	OBJECT		= 3
};


// represent a blob in the image
class   Blob {
public:
	int     id;		/* blob ID        */
	int		x, y;	/* blob position  */
	int		w, h;	/* blob sizes     */
	Class	label;	/* type of blob   */

	Blob (int id, int x, int y, int w, int h);
	Blob (int id, Rect rect, bool invert = false);
};


// handles all functions related to the lists of blobs in one place, amking them easier to use
class BlobList {
public:
	BlobList ();
	BlobList (FillModes fillMode);

	//blob drawing functions
	Mat		paintBlobImage (const Mat& frame, bool labelled);

	//blob extraction functions
	int		extractBlobs (const Mat& fgmask, int connectivity);
	int		removeSmallBlobs (int min_width, int min_height);

	//blob classification functions
	int		classifyBlobs ();

private:
	std::vector<Blob> bloblist;
	FillModes fillMode;
};


//stationary blob extraction functions
int extractStationaryFG (Mat fgmask, Mat& fgmask_history, Mat& sfgmask);


#endif

