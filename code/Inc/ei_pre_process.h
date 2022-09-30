/*
 * ei_pre_process.h
 *
 *  Created on: 27-Sep-2022
 *      Author: Asus
 */

#ifndef EI_PRE_PROCESS_H_
#define EI_PRE_PROCESS_H_

#include "app_x-cube-ai.h"
//#include "model_metadata.h"
#include "model_variables.h"

#define START_BINS    1
#define STOP_BINS     65 //FFT Length/2 + 1 -> FFT length here = 128
#define FFT_LENGTH    128
#define RAW_SAMPLE_COUNT           1250
#define kiss_fft_scalar float
	
int8_t process_impulse(float *input_buf, float *result);
extern float input_test[1250];
#endif /* EI_PRE_PROCESS_H_ */
