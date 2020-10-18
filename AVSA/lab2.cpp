/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 * 
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 */

//system libraries C/C++
#include <stdio.h>
#include <iostream>
#include <sstream>

// opencv libraries
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

// Header ShowManyImages
#include "ShowManyImages.hpp"

// include for blob-related functions
#include "blobs.hpp"

// namespaces
using namespace cv; // avoid using 'cv' to declare OpenCV functions and variables (cv::Mat or Mat)
using namespace std;


#define MIN_WIDTH 20
#define MIN_HEIGHT 30


enum class BackgroundModes {
	stationaryFG = 0,
	noStationaryFG = 1
};


// main function
int main(int argc, char ** argv) 
{
	if (argc != 4) {
		std::cerr << "You need three arguments: FillMode (recursiveFill, sequentialFill, opencvFill), BackgroundMode (noStationaryFG, stationaryFG) and path to dataset (../Lab2/2020AVSALab3_datasets) (the program will look for the following videos: ETRI/ETRI_od_A.avi, PETS2006/PETS2006_S1/PETS2006_S1_C3.mpeg, PETS2006/PETS2006_S4/PETS2006_S4_C3.avi, PETS2006/PETS2006_S5/PETS2006_S5_C3.mpeg, VISOR/visor_Video00.avi, VISOR/visor_Video01.avi, VISOR/visor_Video02.avi, VISOR/visor_Video03.avi)";
		return -1;
	}

	if (argc == 2 && argv[1] == "help") {
		std::cout << "You need three arguments: FillMode (recursiveFill, sequentialFill, opencvFill), BackgroundMode (noStationaryFG, stationaryFG) and path to dataset (../Lab2/2020AVSALab3_datasets) (the program will look for the following videos: ETRI/ETRI_od_A.avi, PETS2006/PETS2006_S1/PETS2006_S1_C3.mpeg, PETS2006/PETS2006_S4/PETS2006_S4_C3.avi, PETS2006/PETS2006_S5/PETS2006_S5_C3.mpeg, VISOR/visor_Video00.avi, VISOR/visor_Video01.avi, VISOR/visor_Video02.avi, VISOR/visor_Video03.avi)\nUsage: lab2 FillMode BackgroundMode videopath";
		return 0;
	}

	BackgroundModes backgroundMode;
	if (string(argv[2]) == "noStationaryFG")
		backgroundMode = BackgroundModes::noStationaryFG;
	else
		backgroundMode = BackgroundModes::stationaryFG;

	FillModes fillMode;
	if (string(argv[1]) == "recursiveFill")
		fillMode = FillModes::recursiveFill;
	else if (argv[1] == "sequentialFill")
		fillMode = FillModes::sequentialFill;
	else
		fillMode = FillModes::opencvFill;

	Mat frame; // current Frame
	Mat fgmask; // foreground mask
	BlobList bloblist (fillMode);

	Mat fgmask_history; // STATIONARY foreground mask
	Mat sfgmask; // STATIONARY foreground mask
	BlobList s_bloblist (fillMode);

	double t, acum_t; //variables for execution time
	double t_freq = getTickFrequency();

	// Paths for the dataset
	string dataset_path = argv[3]; // "../Lab2/2020AVSALab3_datasets";
	string dataset_cat[1] = {""};
	// string baseline_seq[10] = {"AVSS2007/AVSSS07_EASY.mkv","AVSS2007/AVSSS07_HARD.mkv", "ETRI/ETRI_od_a.avi", "PETS2006/PETS2006_S1/PETS2006_S1_C3.mpeg","PETS2006/PETS2006_S4/PETS2006_S4_C3.avi","PETS2006/PETS2006_S5/PETS2006_S5_C3.mpeg","VISOR/visor_Video00.avi","VISOR/visor_Video01.avi","VISOR/visor_Video02.avi","VISOR/visor_Video03.avi"};
	string baseline_seq[8] = {"ETRI/ETRI_od_A.avi","PETS2006/PETS2006_S1/PETS2006_S1_C3.mpeg","PETS2006/PETS2006_S4/PETS2006_S4_C3.avi","PETS2006/PETS2006_S5/PETS2006_S5_C3.mpeg","VISOR/visor_Video00.avi","VISOR/visor_Video01.avi","VISOR/visor_Video02.avi","VISOR/visor_Video03.avi"};
	string image_path = ""; // path to images - this format allows to read consecutive images with filename inXXXXXX.jpq (six digits) starting with 000001


	int NumCat = sizeof(dataset_cat)/sizeof(dataset_cat[0]); // number of categories (have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

	// Loop for all categories
	for (int c=0; c<NumCat; c++ )
	{
		int NumSeq = sizeof(baseline_seq)/sizeof(baseline_seq[0]);  // number of sequences per category ((have faith ... it works! ;) ... each string size is 32 -at leat for the current values-)

		// Loop for all sequence of each category
		for (int s=0; s<NumSeq; s++ )
		{
			VideoCapture cap; // reader to grab videoframes

			// Compose full path of images
			// string inputvideo = dataset_path + "/" + dataset_cat[c] + "/" + baseline_seq[s] + image_path;
			string inputvideo = dataset_path + "/" + baseline_seq[s] + image_path;
			cout << "Accessing sequence at " << inputvideo << endl;

			// open the video file to check if it exists
			cap.open(inputvideo);
			if (!cap.isOpened()) {
				cout << "Could not open video file " << inputvideo << endl;
				return -1;
			}

			// MOG2 approach
			Ptr<BackgroundSubtractor> pMOG2 = cv::createBackgroundSubtractorMOG2();

			// main loop
			Mat img; // current Frame

			int it = 1;
			acum_t = 0;

			for (;;) {

				// get frame
				cap >> img;

				// check if we achieved the end of the file (e.g. img.data is empty)
				if (!img.data)
					break;

				// Time measurement
				t = (double)getTickCount();

				img.copyTo(frame);


				double learningrate;
				// with no stationary foreground detection, the learning rate can be automatic
				if (backgroundMode == BackgroundModes::noStationaryFG)
					learningrate = -1; 
				else // with stationary foreground detection, the learning rate has to be zero, otherwise the stationary objects would get incorporated into th background
					learningrate = 0;

				pMOG2->apply(frame, fgmask, learningrate);
				// 0 bkg, 255 fg, 127 (gray) shadows ...

				int connectivity = 8; // 4 or 8

				// Extract the blobs in fgmask
				bloblist.extractBlobs(fgmask, connectivity);
				bloblist.removeSmallBlobs(MIN_WIDTH, MIN_HEIGHT);

				// Clasify the blobs in fgmask
				bloblist.classifyBlobs();

				if (it == 1) {
					sfgmask = Mat::zeros (Size (fgmask.cols, fgmask.rows), CV_8UC1);
					fgmask_history = Mat::zeros (Size (fgmask.cols, fgmask.rows), CV_32FC1);
				}				

				// STATIONARY BLOBS
				if (backgroundMode == BackgroundModes::stationaryFG)
				{
					// Extract the STATIC blobs in fgmask
					extractStationaryFG (fgmask, fgmask_history, sfgmask);
					s_bloblist.extractBlobs (sfgmask, connectivity);
	
					int min_width = 0;  // to set properly
					int min_height = 0; // to set properly

					s_bloblist.removeSmallBlobs (MIN_WIDTH, MIN_HEIGHT);

					// Clasify the blobs in fgmask
					s_bloblist.classifyBlobs ();
				}				

				// Time measurement
				t = (double)getTickCount() - t;
				acum_t=+t;

				// SHOW RESULTS
				// get the frame number and write it on the current frame

				string title= " | Frame - FgM - Stat FgM | Blobs - Classes - Stat Classes | BlobsFil - ClassesFil - Stat ClassesFil | ("+dataset_cat[c] + "/" + baseline_seq[s] + ")";

				ShowManyImages(title, 6, frame, fgmask, sfgmask,
						bloblist.paintBlobImage(frame, false), bloblist.paintBlobImage(frame, true), s_bloblist.paintBlobImage(frame, true));

				// exit if ESC key is pressed
				if (waitKey (30) == 27) break;
				it++;
			} // main loop

			cout << it - 1 << "frames processed in " << 1000 * acum_t / t_freq << " milliseconds." << endl;

			// release all resources

			cap.release();
			destroyAllWindows();
			waitKey(0); // (should stop till any key is pressed .. doesn't!!!!!)
		}
	}

	return 0;
}




