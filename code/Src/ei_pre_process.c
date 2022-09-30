/*
 * ei_pre_process.c
 *
 *  Created on: 27-Sep-2022
 *      Author: Asus
 */
#include "stdlib.h"
#include "ei_pre_process.h"
#include "arm_math.h"

float spectral_features[EI_CLASSIFIER_NN_INPUT_FRAME_SIZE]= {0};
#define max_var(x, y) \
	(((x) > (y)) ? (x) : (y))

#define max_val(type, x, y) \
	(type)max_var((type)(x), (type)(y))

static int subtract_mean(float *input_buf) {
	// calculate the mean
	float mean;
	arm_mean_f32(input_buf, 1250, &mean);

	// scale by the mean
    for (uint16_t count = 0; count < EI_CLASSIFIER_RAW_SAMPLE_COUNT; count++) {
    	input_buf[count] -= mean;
    }

	return 0;
}

static float fast_sqrt(float x) {
		float temp;
		arm_sqrt_f32(x, &temp);
		return temp;
}    

/**
 * Compute the one-dimensional discrete Fourier Transform for real input.
 * This function computes the one-dimensional n-point discrete Fourier Transform (DFT) of
 * a real-valued array by means of an efficient algorithm called the Fast Fourier Transform (FFT).
 * @param src Source buffer
 * @param src_size Size of the source buffer
 * @param output Output buffer
 * @param output_size Size of the output buffer, should be n_fft / 2 + 1
 * @returns 0 if OK
 */
static int rfft(float *src, size_t src_size, float *output, size_t output_size, size_t n_fft) {
    size_t n_fft_out_features = (n_fft / 2) + 1;
//		float fft_input[128] = {0};
//		float fft_output[65] = {0};  //th

    // truncate if needed
    if (src_size > n_fft) {
        src_size = n_fft;
    }

		float *fft_input = (float*)calloc(128, sizeof(float));
		float *fft_output = (float*)calloc(128, sizeof(float));
    // copy from src to fft_input
    memcpy(fft_input, src, src_size * sizeof(float));

	// hardware acceleration only works for the powers above...
	arm_rfft_fast_instance_f32 rfft_instance;
	int status = arm_rfft_fast_init_f32(&rfft_instance, n_fft);  // look here.
	if (status != ARM_MATH_SUCCESS) {
		return status;
	}

	arm_rfft_fast_f32(&rfft_instance, fft_input, fft_output, 0);

	output[0] = fft_output[0];
	output[n_fft_out_features - 1] = fft_output[1];  //0.320608735

	size_t fft_output_buffer_ix = 2;
	for (size_t ix = 1; ix < n_fft_out_features - 1; ix += 1) {  //n_fft_out_features = 65
		float rms_result;
		arm_rms_f32(fft_output + fft_output_buffer_ix, 2, &rms_result);
		output[ix] = rms_result * fast_sqrt(2);   //         float temp; arm_sqrt_f32(x, &temp); return temp;
		fft_output_buffer_ix += 2;
	}
	
		free(fft_input);
		free(fft_output);

  return 0;
}
/**
  * Power spectrum of a frame
  * @param frame Row of a frame
  * @param frame_size Size of the frame
  * @param out_buffer Out buffer, size should be fft_points
  * @param out_buffer_size Buffer size
  * @param fft_points (int): The length of FFT. If fft_length is greater than frame_len, the frames will be zero-padded.
  * @returns EIDSP_OK if OK
  */
static int power_spectrum(
    float *frame,
    size_t frame_size,
    float *out_buffer,
    size_t out_buffer_size,  //65
    uint16_t fft_points)  //128
{
    int r = rfft(frame, frame_size, out_buffer, out_buffer_size, fft_points);

	  // out_buffer = 0.320608735
    for (size_t ix = 0; ix < out_buffer_size; ix++) {   //out_buffer_size = 65
        out_buffer[ix] = (1.0 / (float)fft_points) * (out_buffer[ix] * out_buffer[ix]);
    }   //second_loop out_buffer = 0.099410668

    return 0;
}

static int welch_max_hold(
        float *input,
        size_t input_size,
        float *output,
        size_t start_bin,
        size_t stop_bin,
        size_t fft_points,
        bool do_overlap)
{
	// save off one point to put back, b/c we're going to calculate in place
	float saved_point;
	bool do_saved_point = false;
	size_t fft_out_size = fft_points / 2 + 1;
	float fft_out[128] = {0};

	// set input as output for in place operation
	//fft_out = input;  //-0.0524538383
	// save off one point to put back, b/c we're going to calculate in place
	saved_point = input[fft_points / 2];  //-0.103853837
	do_saved_point = true;

	// init the output to zeros
	memset(output, 0, sizeof(float) * (stop_bin - start_bin));
	int input_ix = 0;
	while (input_ix < input_size) { //input_size = 1250
		// Figure out if we need any zero padding
		size_t n_input_points = input_ix + fft_points <= input_size ? fft_points
																	: input_size - input_ix;
		power_spectrum(
			input + input_ix,
			n_input_points,
			fft_out,
			(fft_points / 2) + 1,
			fft_points);

		int j = 0;
		// keep the max of the last frame and everything before   //THis is also seems to be working
		for (size_t i = start_bin; i < stop_bin; i++) {
			output[j] = max_val(float, output[j], fft_out[i]);
			j++;
		}

		input_ix += fft_points;  //fft_points = 128
	}

	return 0;
}

static void zero_handling(float *input, size_t input_size)
{
    for (size_t ix = 0; ix < input_size; ix++) {
        if (input[ix] == 0) {
            input[ix] = 1e-10;
        }
    }
}

float fast_log2(float a)
{
    int e;
    float f = frexpf(fabsf(a), &e);
    float y = 1.23149591368684f;
    y *= f;
    y += -4.11852516267426f;
    y *= f;
    y += 6.02197014179219f;
    y *= f;
    y += -3.13396450166353f;
    y += e;
    return y;
}

static int fast_log10(float *input_data)
{
    for (uint8_t ix = 0; ix < 64; ix++) {
        input_data[ix] = fast_log2(input_data[ix]) * 0.3010299956639812f;
    }

    return 0;
}

static int extract_spectral_analysis_features(
        float *input_data,
        float *output_features)  //edited magic happens ere
{
	// Figure bins we remove based on filter cutoff
	size_t num_bins = STOP_BINS - START_BINS;
	float rms = 0;
	subtract_mean(input_data);

	size_t data_size = 1250;  //coulb be replace with macro 1250

	arm_rms_f32(input_data, 1250, &rms);

	output_features[0] = rms;
	output_features++;

	// Standard Deviation
	float stddev = *(output_features-1); //= sqrt(numpy::variance(data_window, data_size));
	// Don't add std dev as a feature b/c it's the same as RMS
	// Skew and Kurtosis w/ shortcut:
	// See definition at https://en.wikipedia.org/wiki/Skewness
	// See definition at https://en.wikipedia.org/wiki/Kurtosis
	// Substitute 0 for mean (b/c it is subtracted out above)
	// Skew becomes: mean(X^3) / stddev^3
	// Kurtosis becomes: mean(X^4) / stddev^4
	// Note, this is the Fisher definition of Kurtosis, so subtract 3
	// (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)
	float s_sum = 0;
	float k_sum = 0;
	float temp;
	for (int i = 0; i < data_size; i++) {
		temp = input_data[i] * input_data[i] * input_data[i];
		s_sum += temp;
		k_sum += temp * input_data[i];
	}
	// Skewness out
	temp = stddev * stddev * stddev;  //0.155815706  //s_sum = 6.0155077
	*output_features++ = (s_sum / data_size) / temp;  //temp = 0.00378297688  //k_sum = 19.7783146
	// Kurtosis out
	*output_features++ = ((k_sum / data_size) / (temp * stddev)) - 3;  //temp = 0.00378297688

	welch_max_hold(input_data, data_size, output_features, START_BINS, STOP_BINS, FFT_LENGTH, false);

	zero_handling(output_features, num_bins);  // output_features = 0.010746059, num_bins=64
	fast_log10(output_features);   //features_out = 0.00378297688

	output_features += num_bins;   //features_out = 7.01770271e-042

	return 0;
}

int8_t process_impulse(float *input_buf, float *result)
{  
	extract_spectral_analysis_features(input_buf, spectral_features);
	return aiRun(spectral_features, result);
}