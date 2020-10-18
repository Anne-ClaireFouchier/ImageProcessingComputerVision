#include "Tracker.hpp"
#include "ShowManyImages.hpp"

using namespace cv;
using namespace std;

// initializing the tracker parameters
Tracker::Tracker (Model* model, unsigned int nrCandidates, unsigned int stride)
{
	this->model = model;
	this->nrCandidates = nrCandidates;
	this->stride = stride;
}

Tracker::~Tracker ()
{
	delete model;
}

// get the best estimate for the given frame
Rect	Tracker::GetEstimate (const Mat& frame) const
{
	// offset the top left corner of the model which will be the starting point for the first candidate
	int offset = (int)(sqrt(nrCandidates) / 2.0 * stride);
	Point start = model->GetLocation ().tl () + Point (-offset, -offset);

	// if the neighbourhood is outside of the frame in the top left corner decrease it to fit into the frame
	if (start.x < 0) start.x = 0;
	if (start.y < 0) start.y = 0;

	// candidates have the same size as the model
	Size candidateSize = model->GetLocation ().size ();
	vector<Rect> candidates;
	vector<double> colorScores;
	vector<double> gradientScores;
	
	// crate a double loop for the top left corners of the candidates
	for (int x = start.x; x < start.x + offset * 2; x += stride) {
		for (int y = start.y; y < start.y + offset * 2; y += stride) {
			Rect currentLoc = Rect (Point (x, y), candidateSize);
			// check if the current candidate is inside the frame
			if (currentLoc.br ().x < frame.cols && currentLoc.br ().y < frame.rows) {
				// calculate the two difference scores
				colorScores.push_back (model->CalculateColorDifference (frame (currentLoc)));
				gradientScores.push_back (model->CalculateGradientDifference (frame (currentLoc)));
				// store the candidates to later select the one belonging to the smallest difference
				candidates.push_back (currentLoc);
			}
		}
	}

	// normalize the scores to be abble to accumulate them
	normalize (colorScores, colorScores, 1, 0, NORM_L1);
	normalize (gradientScores, gradientScores, 1, 0, NORM_L1);

	// add the two normalized scores together
	std::transform (colorScores.begin (), colorScores.end (), gradientScores.begin (),
					colorScores.begin (), std::plus<double> ());

	// find the candidate belonging to the smallest difference
	Rect minLocation = candidates[std::distance (colorScores.begin (), std::min_element (colorScores.begin (), colorScores.end ()))];

	// only for Color Model
	// plot histograms and best candidaets
	//ShowManyImages ("Best candidate", 4,
	//				CreateHistogramImage("Best candidate", model->CalculateFeature (frame (minLocation))),
	//				frame(minLocation),
	//				CreateHistogramImage ("Model", model->GetFeature ()),
	//				model->ConvertPatch(frame(minLocation)));

	// update the model. in the final version only the location is actually updated
	model->UpdateModel (minLocation, frame (minLocation));

	return minLocation;
}
