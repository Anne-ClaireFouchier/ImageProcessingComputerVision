/* Applied Video Sequence Analysis (AVSA)
 *
 *	LAB1.0: Background Subtraction - Unix version
 *	fgesg.cpp
 *
 * 	Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es) & Juan Carlos San Miguel (juancarlos.sanmiguel@uam.es)
 *	VPULab-UAM 2020
 */

#include <opencv2/opencv.hpp>
#include "fgseg.hpp"

using namespace fgseg;

cv::Mat 	accumulate (cv::Mat m) {
	cv::Mat s[3];
	split(m, s);
	for (cv::Mat i : s)
		i = i / 255;

	s[0] = s[0] + s[1] + s[2];
	s[0] = s[0] > 0;
	return s[0];
}

cv::Mat 	triple (cv::Mat m) {
	vector<cv::Mat> t = {m, m, m};
	cv::Mat result;
	cv::merge (t, result);
	return result;
}


//default constructor
bgs::bgs(double threshold, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
	_alpha = 0;
	_selective_update = false;
	_threshold_ghosts = 0;
}

bgs::bgs(double threshold, double alpha, bool selective_bkg_update,bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
	_alpha = alpha;
	_selective_update = selective_bkg_update;
	_threshold_ghosts = 0;

}

bgs::bgs(double threshold, double alpha, bool selective_bkg_update, int threshold_ghosts, bool rgb)
{
	_rgb = rgb;
	_threshold = threshold;
	_alpha = alpha;
	_selective_update = selective_bkg_update;
	_threshold_ghosts = threshold_ghosts;
}

//default destructor
bgs::~bgs(void)
{
}


//method to initialize bkg (first frame - hot start)
void bgs::init_bkg(cv::Mat Frame)
{
	if (_shadowdetection)
			_rgb = true;

	if (!_rgb) {
		cvtColor (Frame, _bkg, cv::COLOR_BGR2GRAY);
		_counter = cv::Mat::zeros(_bkg.size(), CV_8UC1);
		_diff = cv::Mat::zeros(_bkg.size(), CV_8UC1);
	} else {
		Frame.copyTo(_bkg);
		_counter = cv::Mat::zeros(_bkg.size(), CV_8UC3);
		_diff = cv::Mat::zeros(_bkg.size(), CV_8UC3);
	}

	_bgsmask = cv::Mat::zeros(_bkg.size(), CV_8UC1);

	if (_simplegaussian) {
		_bkg.copyTo(_mu);
		_mu.convertTo(_mu, CV_32F);
		cv::Scalar std_init, mean;
		cv::meanStdDev(_bkg, mean, std_init);
		//cout << mean << "\n" << std_init;
		_std = cv::Mat(_bkg.size(), CV_32F, Scalar(std_init[0]));

	}

	if (_multimodal)
			_gmm.init (_bkg, 0.3, 3);
}


//method to perform BackGroundSubtraction
void bgs::bkgSubtraction(cv::Mat Frame)
{
	if (!_rgb){
		cvtColor (Frame, _frame, cv::COLOR_BGR2GRAY);

		if (_simpledetection || _ghosting || _selective_update) {
			// simply take the difference between the background end current frame and create a mask by threesholding the result
			absdiff(_bkg, _frame, _diff);
			_bgsmask = _diff > _threshold;
		}

		if (_selective_update) {
			// create a logical mask, we'll use this to only update those pixels that count as foreground
			cv::Mat bkg_logical = _bgsmask / 255; // 1 is foreground
			// apply the update to the selected pixels
			_bkg = bkg_logical.mul(_alpha*_frame+(1-_alpha)*_bkg) + (1 - bkg_logical).mul(_bkg);
		}

		if (_ghosting) {
			cv::Mat bkg_logical = _bgsmask / 255; // 1 is foreground
			 // increase the counter if it's foreground
			_counter += bkg_logical;
			// zero out counter if it's not foreground
			_counter = _counter.mul (bkg_logical);
			// if the counter reaches the threshold we have to change those pixels
			cv::Mat pixels_to_change = ((_counter > _threshold_ghosts) == 255) / 255;
			// zero out the pixels in background that we're going to update and update and replace them with the ones from the curent frame
			_bkg = _bkg.mul(1 - pixels_to_change);
			_bkg += _frame.mul(pixels_to_change);
			// if we changed a pixel, we zero the counter
			_counter = _counter.mul(1 - pixels_to_change);
		}

		if (_simplegaussian) {
			// need to convert frame to float to to get every result as float
			_frame.convertTo(_frame, CV_32F);
			// calculate the distance between stored means and current frame
			cv::Mat dist = abs(_mu - _frame);
				//cout << "dist: " << sum(dist) << "\n";
			// check if the distance is bigger than a given times the standard deviation
			cv::Mat diff = (dist > _std_coeff * _std);

			// if it's bigger it means it's foreground, so it's the same as background substraction mask
			diff.copyTo(_bgsmask);
				//cout << "bgs: " << sum(_bgsmask / 255) << "\n";
				//cout << "diff: " << sum(diff) << "\n";
			// create a logical (0 or 1) matrix with flaoting values, we'll use this to only update the pixels that are foreground
			diff = diff / 255;
			diff.convertTo(diff, CV_32F);
				//cout << "diff: " << sum(diff) << "\n";
				//cout << type2str(diff.type()) << "\n";
				//cout << "_mu1: " << sum(_mu) << "\n";
				//cout << "_std1: " << sum(_std) << "\n";
			// apply the update on the selected pixels
			_mu = diff.mul(_alpha * _frame + (1 - _alpha) * _mu) + (1.0 - diff).mul(_mu);
				//cv::Mat newstd;
				//sqrt(_alpha * ((_frame - _mu).mul(_frame - _mu)) + (1.0 - _alpha) * _std.mul(_std), newstd);
				//cout << "_std1: " << sum(newstd) << "\n";
				//cout << type2str(newstd.type()) << "\n";
				//_std = diff.mul(newstd) + (1.0 - diff).mul(_std);
			sqrt(diff.mul(_alpha * ((_frame - _mu).mul(_frame - _mu)) + (1.0 - _alpha) * _std.mul(_std)) + (1.0 - diff).mul(_std).mul(_std), _std);
				//cout << "_mu2: " << sum(_mu) << "\n";
				//cout << "_std2: " << sum(_std) << "\n";
		}

		if (_multimodal) {
			_gmm.getFGMask (_frame, _bgsmask);
		}


	} else {
		// almost the same as non-rgb, with small modifications to work with three chanel matrices

		cv::Mat ones(_counter.size(), CV_8UC3, Scalar(1, 1, 1)); // helper matrix

		if (_simpledetection || _ghosting || _selective_update || _shadowdetection) {
			Frame.copyTo(_frame);
			_diff = abs(_frame - _bkg);
			_bgsmask = accumulate (_diff > _threshold);
		}

		if (_selective_update) {
			cv::Mat bkg_logical = triple(_bgsmask / 255);
			cv::Mat bkg_save;
			_bkg.copyTo(bkg_save);
			_bkg = bkg_logical.mul(_alpha*_frame+(1-_alpha)*_bkg) + (ones - bkg_logical).mul(_bkg);
			//_bkg = bkg_save.mul((_bkg==0)/255)+_bkg;
		}

		if (_ghosting) {
			cv::Mat bkg_logical = triple(_bgsmask / 255); // 1 is foreground
			_counter += bkg_logical;
			_counter = _counter.mul (bkg_logical);
			//cout << "counter: " << sum(_counter) << "\n";
			//cout << "bkg: " << sum(bkg_logical) << "\n";
			cv::Mat pixels_to_change = ((_counter > _threshold_ghosts) == 255) / 255;

			_bkg = _bkg.mul(ones - pixels_to_change);
			_bkg += _frame.mul(pixels_to_change);

			_counter = _counter.mul(ones - pixels_to_change);
			//cout << "counter: " << sum(_counter) << "\n";
		}
	}
}



//method to detect and remove shadows in the BGS mask to create FG mask
void bgs::removeShadows()
{
	_bgsmask.copyTo(_shadowmask);

	if (_shadowdetection) {
		// convert background and frame to HSV colorspace and split the 3 chanels
		cv::Mat bkg_hsv, bkg_hsv_split[3];
		cv::Mat frame_hsv, frame_hsv_split[3];
		cvtColor (_bkg, bkg_hsv, cv::COLOR_BGR2HSV);
		cvtColor (_frame, frame_hsv, cv::COLOR_BGR2HSV);
		split (bkg_hsv, bkg_hsv_split);
		split (frame_hsv, frame_hsv_split);

		// first part of the equation, need to convert matrices to floating point to do the division, then we do the comparison
		// with alpha, beta and convert the results back to uchar for further usage (it's a logical result, no need for flaoting points)
		// we reuse the variables along the way to save memory and shorten code lenght
		cv::Mat image_value, background_value;
		frame_hsv_split[2].convertTo(image_value, CV_32F);
		bkg_hsv_split[2].convertTo(background_value, CV_32F);
		image_value = image_value / background_value;
		background_value = image_value < b;
		image_value = image_value > a;
		image_value.convertTo(image_value, CV_8UC1);
		background_value.convertTo(background_value, CV_8UC1);

		// calculate the three parts of the consition separatly and logically and them together
		cv::Mat value = ((image_value / 255 + background_value / 255) > 1) / 255;
		cv::Mat saturation = (abs (frame_hsv_split[1] - bkg_hsv_split[1]) < tau_s) / 255;
		cv::Mat hue = (cv::min (cv::Mat (abs (frame_hsv_split[0] - bkg_hsv_split[0])), cv::Mat (360 - abs (frame_hsv_split[0] - bkg_hsv_split[0]))) < tau_h) / 255;
		cv::Mat shadow = ((value + saturation + hue) == 3) / 255;
		//cout << "value: " << sum(value) << "\n";
		//cout << "saturation: " << sum(saturation) << "\n";
		//cout << "hue: " << sum(hue) << "\n";
		//cout << "shadow: " << sum(shadow) << "\n";

		// we take only those pixels that were indentified as foreground
		_shadowmask = (shadow.mul (_bgsmask / 255)) * 255;
	} else {
		absdiff (_bgsmask, _bgsmask, _shadowmask);
	}

	absdiff(_bgsmask, _shadowmask, _fgmask); // eliminates shadows from bgsmask
}


//ADD ADDITIONAL FUNCTIONS HERE




