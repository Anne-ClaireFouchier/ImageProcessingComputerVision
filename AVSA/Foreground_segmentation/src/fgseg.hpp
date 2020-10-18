/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.hpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */


#include <opencv2/opencv.hpp>
#include "gmm.hpp"

#ifndef FGSEG_H_INCLUDE
#define FGSEG_H_INCLUDE

using namespace cv;
using namespace std;

namespace fgseg {


	//Declaration of FGSeg class based on BackGround Subtraction (bgs)
	class bgs{
	public:

		//constructor with parameter "threshold"
		bgs(double threshold, bool rgb);
		bgs(double threshold, double alpha, bool selective_bkg_update, bool rgb);
		bgs(double threshold, double alpha, bool selective_bkg_update, int threshold_ghosts, bool rgb);

		//destructor
		~bgs(void);

		//method to initialize bkg (first frame - hot start)
		void init_bkg(cv::Mat Frame);

		//method to perform BackGroundSubtraction
		void bkgSubtraction(cv::Mat Frame);

		//method to detect and remove shadows in the binary BGS mask
		void removeShadows();

		//returns the BG image
		cv::Mat getBG(){return _bkg;};

		//returns the DIFF image
		cv::Mat getDiff(){return _diff;};

		//returns the BGS mask
		cv::Mat getBGSmask(){return _bgsmask;};

		//returns the binary mask with detected shadows
		cv::Mat getShadowMask(){return _shadowmask;};

		//returns the binary FG mask
		cv::Mat getFGmask(){return _fgmask;};


		//ADD ADITIONAL METHODS HERE
		//...
	private:
		cv::Mat _bkg; //Background model
		cv::Mat	_frame; //current frame
		cv::Mat _diff; //abs diff frame
		cv::Mat _bgsmask; //binary image for bgssub (FG)
		cv::Mat _shadowmask; //binary mask for detected shadows
		cv::Mat _fgmask; //binary image for foreground (FG)
		cv::Mat	_counter; // counter for ghost supression
		cv::Mat _mu; // mean values for pixels
		cv::Mat _std; // standard deviation for pixels
		GMM _gmm;

		bool _rgb;
		double _threshold;
		double _alpha;
		bool _selective_update;
		int _threshold_ghosts;
		float _std_coeff = 1.0;

		bool _ghosting = false;
		bool _shadowdetection = false;
		bool _simpledetection = false;
		bool _simplegaussian = false;
		bool _multimodal = true;

		// coefficient for shadow detection
		double a = 0.3, b = 0.9;
		double tau_s = 80, tau_h = 70;

		//ADD ADITIONAL VARIABLES HERE
		//...

	};//end of class bgs

}//end of namespace

#endif




