/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Target ISA:  ARMv7E-M
 * Reference papers:
 * 	- MCUNet: Tiny Deep Learning on IoT Device, NIPS 2020
 *	- MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning, NIPS 2021
 * Contact author:
 *  - Wei-Chen Wang, wweichen@mit.edu
 * 	- Wei-Ming Chen, wmchen@mit.edu
 * 	- Ji Lin, jilin@mit.edu
 * 	- Song Han, songhan@mit.edu
 * -------------------------------------------------------------------- */
#include "arm_nnfunctions.h"
#include "img2col_element.h"
#include "tinyengine_function.h"
#include "macro.h"

#define DIM_KER_X (1U)
#define DIM_KER_Y (1U)
#define STRIDE (1U)
#define PAD (0U)

tinyengine_status convolve_1x1_s8_fpreq(const q7_t *input,
		const uint16_t input_x, const uint16_t input_y, const uint16_t input_ch,
		const q7_t *kernel, const int32_t *bias, const float *scales,
		const int32_t out_offset, const int32_t input_offset,
		const int32_t out_activation_min, const int32_t out_activation_max,
		q7_t *output, const uint16_t output_x, const uint16_t output_y,
		const uint16_t output_ch, q15_t *runtime_buf) 
{
	if (input_ch % 4 != 0 || input_ch % 2 != 0) {
		return PARAM_NO_SUPPORT;
	}

#if (!LOOP_REORDERING)
	/* This part is a TFLite-like implementation */
	for (int out_y = 0; out_y < output_y; ++out_y) {
		for (int out_x = 0; out_x < output_x; ++out_x) {
			for (int out_channel = 0; out_channel < output_ch; ++out_channel) {
				int32_t sum = 0;
				for (int filter_y = 0; filter_y < DIM_KER_Y; ++filter_y) {
					for (int filter_x = 0; filter_x < DIM_KER_X; ++filter_x) {
						for (int in_channel = 0; in_channel < input_ch; ++in_channel) {
							int32_t input_val = input[OffsetC(input_y, input_x, input_ch, 0, out_y, out_x, in_channel)];
							int32_t filter_val = kernel[OffsetC(DIM_KER_Y, DIM_KER_X, input_ch, out_channel, filter_y, filter_x, in_channel)];
							sum += filter_val * (input_val + input_offset);
						}
					}
				}

				if (bias) {
					sum += bias[out_channel];
				}
				sum = (int32_t) ((float)sum * scales[out_channel]);
				sum += out_offset;
				sum = MAX(sum, out_activation_min);
				sum = MIN(sum, out_activation_max);

				output[OffsetC(output_y, output_x, output_ch, 0, out_y, out_x, out_channel)] = (int8_t)(sum);
			}
		}
	}
#else  // if (LOOP_REORDERING)

	int32_t i_element;
	const int32_t num_elements = output_x * output_y;
	(void) input_x;
	(void) input_y;

	q7_t *input_start = input;
	const q7_t *kernel_start = kernel;
	q7_t *out = output;

#if (!LOOP_UNROLLING)
	for (i_element = 0; i_element < num_elements; i_element++) {
		kernel = kernel_start;
		for (int out_channel = 0; out_channel < output_ch; ++out_channel) {
			int32_t sum = 0;
			input = input_start;

			for (int in_channel = 0; in_channel < input_ch; ++in_channel) {
				int32_t input_val = *input++;
				int32_t filter_val = *kernel++;
				sum += filter_val * (input_val + input_offset);
			}

			if (bias) {
				sum += bias[out_channel];
			}
			sum = (int32_t) ((float)sum * scales[out_channel]);
			sum += out_offset;
			sum = MAX(sum, out_activation_min);
			sum = MIN(sum, out_activation_max);

			*out++ = (int8_t)(sum);
		}
		input_start += input_ch;
	}

#else  // if (LOOP_UNROLLING)
#if (!SIMD)
	for (i_element = 0; i_element < num_elements / 2; i_element++) {
		q7_t *kernel_0 = kernel_start;
		q7_t *kernel_1 = kernel_start + input_ch;
		q7_t *out_0 = output;
		q7_t *out_1 = output + output_ch;

		for (int out_channel = 0; out_channel < output_ch / 2; ++out_channel) {
			int32_t sum_0 = 0;
			int32_t sum_1 = 0;
			int32_t sum_2 = 0;
			int32_t sum_3 = 0;
			q7_t *input_0 = input_start;
			q7_t *input_1 = input_start + input_ch;

			for (int in_channel = 0; in_channel < input_ch / 4; ++in_channel) {
				int32_t input_val = *input_0;
				int32_t filter_val = *kernel_0;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_1;
				filter_val = *kernel_0++;
				sum_1 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1;
				sum_2 += filter_val * (input_val + input_offset);
				input_val = *input_1++;
				filter_val = *kernel_1++;
				sum_3 += filter_val * (input_val + input_offset);

				input_val = *input_0;
				filter_val = *kernel_0;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_1;
				filter_val = *kernel_0++;
				sum_1 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1;
				sum_2 += filter_val * (input_val + input_offset);
				input_val = *input_1++;
				filter_val = *kernel_1++;
				sum_3 += filter_val * (input_val + input_offset);

				input_val = *input_0;
				filter_val = *kernel_0;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_1;
				filter_val = *kernel_0++;
				sum_1 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1;
				sum_2 += filter_val * (input_val + input_offset);
				input_val = *input_1++;
				filter_val = *kernel_1++;
				sum_3 += filter_val * (input_val + input_offset);

				input_val = *input_0;
				filter_val = *kernel_0;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_1;
				filter_val = *kernel_0++;
				sum_1 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1;
				sum_2 += filter_val * (input_val + input_offset);
				input_val = *input_1++;
				filter_val = *kernel_1++;
				sum_3 += filter_val * (input_val + input_offset);
			}

			if (bias) {
				sum_0 += bias[out_channel * 2];
				sum_1 += bias[out_channel * 2];
				sum_2 += bias[out_channel * 2 + 1];
				sum_3 += bias[out_channel * 2 + 1];
			}

			sum_0 = (int32_t) ((float)sum_0 * scales[out_channel * 2]);
			sum_0 += out_offset;
			sum_0 = MAX(sum_0, out_activation_min);
			sum_0 = MIN(sum_0, out_activation_max);
			*out_0++ = (int8_t)(sum_0);

			sum_1 = (int32_t) ((float)sum_1 * scales[out_channel * 2]);
			sum_1 += out_offset;
			sum_1 = MAX(sum_1, out_activation_min);
			sum_1 = MIN(sum_1, out_activation_max);
			*out_1++ = (int8_t)(sum_1);

			sum_2 = (int32_t) ((float)sum_2 * scales[out_channel * 2 + 1]);
			sum_2 += out_offset;
			sum_2 = MAX(sum_2, out_activation_min);
			sum_2 = MIN(sum_2, out_activation_max);
			*out_0++ = (int8_t)(sum_2);

			sum_3 = (int32_t) ((float)sum_3 * scales[out_channel * 2 + 1]);
			sum_3 += out_offset;
			sum_3 = MAX(sum_3, out_activation_min);
			sum_3 = MIN(sum_3, out_activation_max);
			*out_1++ = (int8_t)(sum_3);

			kernel_0 += input_ch;
			kernel_1 += input_ch;
		}

		input_start += input_ch * 2;
		output += output_ch * 2;
	}

	/* check if there is an odd column left-over for computation */
	if (num_elements & 0x1) {
		q7_t *kernel_0 = kernel_start;
		q7_t *kernel_1 = kernel_start + input_ch;
		q7_t *out_0 = output;

		for (int out_channel = 0; out_channel < output_ch / 2; ++out_channel) {
			int32_t sum_0 = 0;
			int32_t sum_1 = 0;
			q7_t *input_0 = input_start;

			for (int in_channel = 0; in_channel < input_ch / 4; ++in_channel) {
				int32_t input_val = *input_0;
				int32_t filter_val = *kernel_0++;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1++;
				sum_1 += filter_val * (input_val + input_offset);

				input_val = *input_0;
				filter_val = *kernel_0++;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1++;
				sum_1 += filter_val * (input_val + input_offset);

				input_val = *input_0;
				filter_val = *kernel_0++;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1++;
				sum_1 += filter_val * (input_val + input_offset);

				input_val = *input_0;
				filter_val = *kernel_0++;
				sum_0 += filter_val * (input_val + input_offset);
				input_val = *input_0++;
				filter_val = *kernel_1++;
				sum_1 += filter_val * (input_val + input_offset);
			}

			if (bias) {
				sum_0 += bias[out_channel * 2];
				sum_1 += bias[out_channel * 2 + 1];
			}

			sum_0 = (int32_t) ((float)sum_0 * scales[out_channel * 2]);
			sum_0 += out_offset;
			sum_0 = MAX(sum_0, out_activation_min);
			sum_0 = MIN(sum_0, out_activation_max);
			*out_0++ = (int8_t)(sum_0);

			sum_1 = (int32_t) ((float)sum_1 * scales[out_channel * 2 + 1]);
			sum_1 += out_offset;
			sum_1 = MAX(sum_1, out_activation_min);
			sum_1 = MIN(sum_1, out_activation_max);
			*out_0++ = (int8_t)(sum_1);

			kernel_0 += input_ch;
			kernel_1 += input_ch;
		}
	}

#else  // if (SIMD)
	/* Partial(two columns) im2col buffer */
	q15_t *two_column_buffer = runtime_buf;

	const int channel_div4 = (input_ch >> 2);
	const int16_t inoff16 = input_offset;
	q31_t offset_q15x2 = __PKHBT(inoff16, inoff16, 16);

	for (i_element = 0; i_element < num_elements / 2; i_element++) {
		/* Fill buffer for partial im2col - two columns at a time */
		q7_t *src = &input[i_element * input_ch * 2];

		q15_t *dst = two_column_buffer;

		//use variables
		q31_t in_q7x4;
		q31_t in_q15x2_1;
		q31_t in_q15x2_2;
		q31_t out_q15x2_1;
		q31_t out_q15x2_2;

		int cnt = channel_div4;	//two columns

		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

		out = mat_mult_kernel_s8_s16_reordered_fpreq(kernel, two_column_buffer,
				output_ch, scales, (q7_t) out_offset, out_activation_min,
				out_activation_max, input_ch * DIM_KER_Y * DIM_KER_X, bias,
				out);
	}

	/* check if there is an odd column left-over for computation */
	if (num_elements & 0x1) {
		int32_t i_ch_out;
		const q7_t *ker_a = kernel;
		q7_t *src = &input[(num_elements - 1) * input_ch];
		q15_t *dst = two_column_buffer;

		//use variables
		q31_t in_q7x4;
		q31_t in_q15x2_1;
		q31_t in_q15x2_2;
		q31_t out_q15x2_1;
		q31_t out_q15x2_2;

		int cnt = channel_div4;	//two * numof2col columns
		while (cnt > 0) {
			q7_q15_offset_reordered_ele(src, dst)
			cnt--;
		}

		for (i_ch_out = 0; i_ch_out < output_ch; i_ch_out++) {
			q31_t sum = bias[i_ch_out];

			/* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
			const q15_t *ip_as_col = runtime_buf;
			uint16_t col_count = (input_ch * DIM_KER_X * DIM_KER_Y) >> 2;

			while (col_count) {
				q31_t ker_a1, ker_a2;
				q31_t in_b1, in_b2;
				ker_a = read_and_pad_reordered(ker_a, &ker_a1, &ker_a2);

				in_b1 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a1, in_b1, sum);
				in_b2 = arm_nn_read_q15x2_ia(&ip_as_col);
				sum = __SMLAD(ker_a2, in_b2, sum);

				col_count--;
			}

			sum = (q31_t) ((float) sum * scales[i_ch_out]);
			sum += out_offset;
			sum = MAX(sum, out_activation_min);
			sum = MIN(sum, out_activation_max);
			*out++ = (q7_t) sum;
		}
	}
#endif  // end of (!SIMD)

#endif  // end of (!LOOP_UNROLLING)

#endif  // end of (!LOOP_REORDERING)

	/* Return to application */
	return STATE_SUCCESS;
}
