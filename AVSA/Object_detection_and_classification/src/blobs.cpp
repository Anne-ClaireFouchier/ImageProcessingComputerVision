#include "blobs.hpp"


Blob::Blob (int id, int x, int y, int w, int h) : id (id), x (x), y (y), w (w), h (h), label (Class (0))
{
}


Blob::Blob (int id, Rect rect, bool invert /*= false*/)
{
	// in some cases there were inverted rectangles, so this way we could easily flip them
	if (invert) {
		*this = Blob (id, rect.y, rect.x, rect.height, rect.width);
	}
	else {
		*this = Blob (id, rect.x, rect.y, rect.width, rect.height);
	}
}


BlobList::BlobList () : fillMode (FillModes::recursiveFill)
{
}


BlobList::BlobList (FillModes fillMode) : fillMode (fillMode)
{
}


/**
 *	Draws blobs with different rectangles on the image 'frame'. All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param pBlobList List to store the blobs found
 * \param labelled - true write label and color bb, false does not write label nor color bb
 *
 * \return Image containing the draw blobs. If no blobs have to be painted
 *  or arguments are wrong, the function returns a copy of the original "frame".
 *
 */

Mat		BlobList::paintBlobImage (const cv::Mat& frame, bool labelled)
{
	cv::Mat blobImage;
	frame.copyTo (blobImage);

	//paint each blob of the list
	for (Blob blob : bloblist) {
		Scalar color;
		std::string label = "";
		switch (blob.label) {
			case Class::PERSON:
				color = Scalar (255, 0, 0);
				label = "PERSON";
				break;
			case Class::CAR:
				color = Scalar (0, 255, 0);
				label = "CAR";
				break;
			case Class::OBJECT:
				color = Scalar (0, 0, 255);
				label = "OBJECT";
				break;
			default:
				color = Scalar (255, 255, 255);
				label = "UNKOWN";
		}

		Point p1 = Point (blob.x, blob.y);
		Point p2 = Point (blob.x + blob.w, blob.y + blob.h);

		rectangle (blobImage, p1, p2, color, 1, 8, 0);

		if (labelled) {
			rectangle (blobImage, p1, p2, color, 1, 8, 0);
			putText (blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
		}
		else {
			rectangle (blobImage, p1, p2, Scalar (255, 255, 255), 1, 8, 0);
		}
	}

	return blobImage;
}


// used for traversing the neighbours of a pixel
const int	dx[8] = { -1, 0, 1,  0, -1, -1,  1, 1 };
const int	dy[8] = { 0, 1, 0, -1, -1,  1, -1, 1 };
const int	int_max = std::numeric_limits<int>::max ();


// finds the bounding box belonging to the blob filled with 'value'
// deprecated
Rect	findBoundingBox (Mat& input, int value)
{
	std::vector<int> x_coord;
	std::vector<int> y_coord;

	// store the coordinates of the image containing the given value, meaning the blob
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (input.at<int> (i, j) == value) {
				x_coord.push_back (j);
				y_coord.push_back (i);
			}
		}
	}

	// if the vector is empty we didn't find any pixel with the given value, 
	if (x_coord.empty ())
		return Rect (0, 0, 0, 0);

	// search for the min and max elements of the vector, meaning the smallest and biggest coordinates
	// these represent the edges of the area of the blob
	auto x_minmax = std::minmax_element (x_coord.begin (), x_coord.end ());
	auto y_minmax = std::minmax_element (y_coord.begin (), y_coord.end ());

	// using the the found coordinates we can determine the top-left corner, width and height
	return Rect (*x_minmax.first, *y_minmax.first, *x_minmax.second - *x_minmax.first, *y_minmax.second - *y_minmax.first);
}


// helper function to update the rect to include the given point too
void	updateRect (const Point point, Rect& rect)
{
	if (point.x < rect.x)
		rect.x = point.x;
	else if (point.x > rect.x + rect.width)
		rect.width = point.x - rect.x;

	if (point.y < rect.y)
		rect.y = point.y;
	else if (point.y > rect.y + rect.height)
		rect.height = point.y - rect.y;
}


// does the recursive step of the fillRecursive method
void	doFillRecursive (Mat& input, const Point& pixel, const int& connectivity, const int& value, Rect& rect)
{
	// set the current pixel to 'value', we only call this function on a pixel that is foreground, so we are sure that we can set this to value
	input.at<int> (pixel) = value;

	// update the rect so it would include the current point too
	updateRect (pixel, rect);

	// a for loop with the preset offsets is used to visit all the neighbours of the pixel
	for (size_t i = 0; i < connectivity; i++) {
		// with the offset we get the neighbour pixel
		Point new_p = Point (pixel.x + dx[i], pixel.y + dy[i]);
		
		// if the neighbour pixel coordinates are not ouside of the bounds of the image and it's a foreground we call again this method with the 
		// neighbour pixel's coordinates
		if (new_p.x >= 0 && new_p.x < input.cols && new_p.y >= 0 && new_p.y < input.rows && input.at<int> (new_p) == 255)
			doFillRecursive (input, new_p, connectivity, value, rect);
	}
}


// a wrapper for doFillRecursive, it resets the rectangle and does the recursive step. It is the recursive implementation of Grassfire
void	fillRecursive (Mat& input, const Point& pixel, int connectivity, int value, Rect& rect)
{
	// we don't know what kind of rect we get, so we have to reset it to start from the seed point
	rect = Rect (pixel.x, pixel.y, 0, 0);
	doFillRecursive (input, pixel, connectivity, value, rect);
}


// sequential implementatin for the Grassfire algorithm
void	fillSequential (Mat& input, const Point& pixel, int connectivity, int value, Rect& rect)
{
	// a FIFO queue is used to store the pixel that ahve to be visited
	std::queue<Point> q;
	// the starting pixel is added to the queue
	q.push (pixel);

	// we don't know what kind of rect we get, so we have to reset it to start from the seed point
	rect = Rect (pixel.x, pixel.y, 0, 0);

	// while we still have pixels to visit in the queue we will repeat the steps
	while (!q.empty ()) {
		// the first element from the queue is removed, we will process this pixel at this step
		Point p = q.front ();
		q.pop ();

		// if the pixel is not a foreground we don't do anything. Unlike with the recursive implementation, here it can happen the the pixel in the queue
		// is already processed, so not checking this might even leed to an infinite loop
		if (input.at<int> (p) != 255)
			continue;

		// set the pixel to 'value' now that we know it is a foreground pixel
		input.at<int> (p) = value;

		updateRect (p, rect);

		// visit all the neighbours with the use of the offset vector like in the recursive implementation
		for (size_t i = 0; i < connectivity; i++) {
			// the neighbpur pixel coords
			Point new_p = Point (p.x + dx[i], p.y + dy[i]);

			// if the neighbour is foreground we add it to the queue to be visited
			if (new_p.x >= 0 && new_p.x < input.cols && new_p.y >= 0 && new_p.y < input.rows && input.at<int> (new_p) == 255)
				q.push (new_p);
		}
	}
}


/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation)
 */

int		BlobList::extractBlobs (const cv::Mat& fgmask, int connectivity)
{
	// check if connectivity has a valid value
	if (connectivity != 4 && connectivity != 8)
		return -1;

	Mat aux; // image to be updated each time a blob is detected (blob cleared)
	fgmask.convertTo (aux, CV_32SC1);

	//clear blob list (to fill with this function)
	bloblist.clear ();

	//Connected component analysis
	// counter is used as the value to fill the given blob. starts from int_max and decreases, this way these values will be unique in the matrix 
	int counter = int_max;
	Rect rect;

	// traverse the image looking for foreground pixels
	for (int i = 0; i < aux.rows; i++) {
		for (int j = 0; j < aux.cols; j++) {
			// if a foreground pixel is found we call one of the fill methods to fill the blob with the given value and determine
			// the corresponding bounding box
			if (aux.at<int> (i, j) == 255) {
				// recursive Grassfire implementation
				if (fillMode == FillModes::recursiveFill)
					fillRecursive (aux, Point (j, i), connectivity, counter, rect);

				// sequential Grassfire implementation
				if (fillMode == FillModes::sequentialFill)
					fillSequential (aux, Point (j, i), connectivity, counter, rect);

				// usign OpenCV function
				if (fillMode == FillModes::opencvFill)
					cv::floodFill (aux, Point (j, i), counter, &rect, Scalar (), Scalar (), connectivity);
				
				// the found bounding box is added to the bloblist
				bloblist.push_back (Blob (i, rect));
				// the counter is decreased, that will be the fill value for the next pixel
				counter--;
			}
		}
	}

	return 1;
}

// removes the blobs smaller than a given value from the lsit
int		BlobList::removeSmallBlobs (int min_width, int min_height)
{
	// determines the elements of the list for which the lambda function return false and moves them to the end of the list
	auto newEnd = std::remove_if (bloblist.begin (), bloblist.end (), [&] (Blob blob)
	{
		return blob.w < min_width || blob.h < min_height;
	});

	// removes the previously determined elements from the bloblist
	bloblist.erase (newEnd, bloblist.end ());

	return 1;
}


/**
 *	Blob classification between the available classes in 'Blob.hpp' (see CLASS typedef). All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation)
 */

 // ASPECT RATIO MODELS
#define MEAN_PERSON 0.3950
#define STD_PERSON 0.1887

#define MEAN_CAR 1.4736
#define STD_CAR 0.2329

#define MEAN_OBJECT 1.2111
#define STD_OBJECT 0.4470

// helper vector, so we can handle the three classes similarly
const	std::vector<std::pair<double, double>> classes = { {MEAN_PERSON, STD_PERSON}, {MEAN_CAR, STD_CAR}, {MEAN_OBJECT, STD_OBJECT} };

// end ASPECT RATIO MODELS

float	ED (float val1, float val2)
{
	return sqrt (pow (val1 - val2, 2));
}


int		BlobList::classifyBlobs ()
{
	//classify each blob of the list
	for (Blob& blob : bloblist) {
		// we will gather the current blobs feature distance from all the models
		std::vector<double> distances;
		// this represent the Unknown class, it's int_max - 1, because int_max value will mean an invalid distance, so int_max - 1 is the highest 
		// distance value (furthest) so if all the other values are int_max the min search will still find this value and assign Unknown to the class
		distances.push_back (int_max - 1);

		// goes through the mean, std pairs of the classes
		for (std::pair<double, double> c : classes) {
			// calculate the distance from the model
			//double dist = abs (blob.w / blob.h - c.first);
			double dist = abs ((double)blob.w / (double)blob.h - c.first);
			// if the distance falls into the [mean - std, mean + std] interval it's a valid distance, we add it to the distance vector
			if (dist <= 3.0 * c.second)
				distances.push_back (dist);
			else // if it's outside the interval we consider it as invalid, it can't be this class, we will add int_max to the vector instead
				distances.push_back (int_max);
		}

		// the argmin of the distance vector is determined. 
		int argMin = std::distance (distances.begin (), std::min_element (distances.begin (), distances.end ()));

		// Since we know the order of the classes (Class enum) and we checked the distances in the same order
		// (paying attention to the Unknown class' place) this argmin will be the index of the corresponding class in the enum
		blob.label = Class (argMin);
	}

	return 1;
}


//stationary blob extraction function
 /**
  *	Stationary FG detection
  *
  * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
  * \param fgmask_history Foreground history counter image (1-channel integer image)
  * \param sfgmask Foreground/Background segmentation mask (1-channel binary image)
  *
  * \return Operation code (negative if not succesfull operation)
  *
  *
  * Based on: Stationary foreground detection for video-surveillance based on foreground and motion history images, D.Ortego, J.C.SanMiguel, AVSS2013
  *
  */

constexpr int		FPS = 25; //check in video - not really critical
constexpr int		SECS_STATIONARY = 1; // to set;
constexpr float		I_COST = 3; // to set // increment cost for stationarity detection
constexpr float		D_COST = 2; // to set // decrement cost for stationarity detection
constexpr float		STAT_TH = 0.5; // to set


int		extractStationaryFG (Mat fgmask, Mat& fgmask_history, Mat& sfgmask)
{
	int numframes4static = (int)(FPS * SECS_STATIONARY);
	// the shadow are eliminated from the mask
	fgmask = fgmask == 255;
	// the fgmask is converted to float too, so all the matrices would have the same type
	fgmask.convertTo(fgmask, CV_32FC1);
	// the update step with the corresponding weights on the background and foreground pixels is done
	fgmask_history = fgmask_history + I_COST * (fgmask / 255) - D_COST * (1 - fgmask / 255);
	// a helper matrix
	Mat ones = Mat::ones(fgmask_history.size(), CV_32FC1);
	// the temporary normalized version of the the history mask is crated
	Mat fgmask_history_norm = min (ones, fgmask_history / numframes4static);
	// this history mask is thresholded and this determines the stationary foreground mask
	sfgmask = fgmask_history_norm > STAT_TH;

	return 1;
}




