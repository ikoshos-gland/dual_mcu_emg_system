/**
 * @file emg_processing.c
 * @brief EMG Signal Processing and Feature Extraction Implementation
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include "emg_processing.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* Private defines -----------------------------------------------------------*/
#define EMG_FILTER_SETTLING_TIME    100     // Filter settling time in samples
#define EMG_FEATURE_BUFFER_SIZE     10      // Number of feature vectors to buffer
#define EMG_NORMALIZATION_SAMPLES   1000    // Samples for normalization parameter update

/* Private variables ---------------------------------------------------------*/
EMG_ProcessingTypeDef emg_processing;
static float32_t temp_buffer[EMG_WINDOW_SIZE];
static float32_t filtered_buffer[EMG_WINDOW_SIZE];

/* Private function prototypes -----------------------------------------------*/
static EMG_StatusTypeDef EMG_InitializeFilters(EMG_ProcessingTypeDef *hemg);
static EMG_StatusTypeDef EMG_InitializeWindows(EMG_ProcessingTypeDef *hemg);
static EMG_StatusTypeDef EMG_ProcessChannel(EMG_ProcessingTypeDef *hemg, uint8_t channel);
static void EMG_UpdateFeatureBuffer(EMG_ProcessingTypeDef *hemg);

/* Filter coefficient tables */
static const float32_t bandpass_coeffs_b[] = {
    0.0007820, 0.0000000, -0.0015640, 0.0000000, 0.0007820
};
static const float32_t bandpass_coeffs_a[] = {
    1.0000000, -3.4277216, 4.4526782, -2.5795082, 0.5547866
};

static const float32_t notch_coeffs_b[] = {
    0.9565436, -3.3948856, 5.3473984, -3.3948856, 0.9565436
};
static const float32_t notch_coeffs_a[] = {
    1.0000000, -3.3948856, 5.3049420, -3.3948856, 0.9565436
};

/* Public functions ----------------------------------------------------------*/

/**
 * @brief Initialize EMG processing
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_Processing_Init(EMG_ProcessingTypeDef *hemg)
{
    if (hemg == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Initialize buffers */
    memset(hemg->input_buffer, 0, sizeof(hemg->input_buffer));
    hemg->buffer_head = 0;
    hemg->buffer_tail = 0;
    
    /* Initialize processing windows */
    status = EMG_InitializeWindows(hemg);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Initialize filters */
    status = EMG_InitializeFilters(hemg);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Initialize FFT */
    status = EMG_FFT_Init(hemg);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Initialize feature vectors */
    memset(&hemg->current_features, 0, sizeof(hemg->current_features));
    memset(&hemg->previous_features, 0, sizeof(hemg->previous_features));
    
    /* Initialize normalization parameters */
    for (uint16_t i = 0; i < FEATURE_VECTOR_SIZE; i++) {
        hemg->feature_min[i] = 1000.0f;
        hemg->feature_max[i] = -1000.0f;
        hemg->feature_mean[i] = 0.0f;
        hemg->feature_std[i] = 1.0f;
    }
    
    hemg->sample_count = 0;
    hemg->feature_count = 0;
    hemg->is_initialized = true;
    hemg->is_running = false;
    
    return EMG_OK;
}

/**
 * @brief De-initialize EMG processing
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_Processing_DeInit(EMG_ProcessingTypeDef *hemg)
{
    if (hemg == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Stop processing if running */
    if (hemg->is_running) {
        EMG_Processing_Stop(hemg);
    }
    
    /* Clean up windows */
    for (uint8_t i = 0; i < EMG_CHANNELS; i++) {
        if (hemg->windows[i].data != NULL) {
            free(hemg->windows[i].data);
            hemg->windows[i].data = NULL;
        }
    }
    
    hemg->is_initialized = false;
    
    return EMG_OK;
}

/**
 * @brief Start EMG processing
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_Processing_Start(EMG_ProcessingTypeDef *hemg)
{
    if (hemg == NULL || !hemg->is_initialized) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Reset filters */
    for (uint8_t i = 0; i < EMG_CHANNELS; i++) {
        EMG_Filter_Reset(&hemg->bandpass_filter[i]);
        EMG_Filter_Reset(&hemg->notch_filter[i]);
        EMG_Filter_Reset(&hemg->comb_filter[i]);
    }
    
    /* Reset counters */
    hemg->sample_count = 0;
    hemg->feature_count = 0;
    hemg->buffer_head = 0;
    hemg->buffer_tail = 0;
    
    hemg->is_running = true;
    
    return EMG_OK;
}

/**
 * @brief Stop EMG processing
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_Processing_Stop(EMG_ProcessingTypeDef *hemg)
{
    if (hemg == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    hemg->is_running = false;
    
    return EMG_OK;
}

/**
 * @brief Process a single EMG sample
 * @param hemg: EMG processing handle pointer
 * @param data: ADS1299 data structure
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_ProcessSample(EMG_ProcessingTypeDef *hemg, ADS1299_DataTypeDef *data)
{
    if (hemg == NULL || data == NULL || !hemg->is_running) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Convert and store samples */
    for (uint8_t channel = 0; channel < EMG_CHANNELS; channel++) {
        /* Convert to float and scale to microvolts */
        float32_t sample = ADS1299_ConvertToVoltage(data->channel_data[channel], ADS1299_CH_GAIN_12);
        
        /* Store in input buffer */
        hemg->input_buffer[channel][hemg->buffer_head] = sample;
        
        /* Add to processing window */
        EMG_Window_AddSample(&hemg->windows[channel], sample);
        
        /* Check if window is ready for processing */
        if (EMG_Window_IsReady(&hemg->windows[channel])) {
            status = EMG_ProcessChannel(hemg, channel);
            if (status != EMG_OK) {
                return status;
            }
        }
    }
    
    /* Update buffer pointers */
    hemg->buffer_head = (hemg->buffer_head + 1) % EMG_BUFFER_SIZE;
    hemg->sample_count++;
    
    return EMG_OK;
}

/**
 * @brief Process a single channel
 * @param hemg: EMG processing handle pointer
 * @param channel: Channel number
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef EMG_ProcessChannel(EMG_ProcessingTypeDef *hemg, uint8_t channel)
{
    if (hemg == NULL || channel >= EMG_CHANNELS) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Get window data */
    status = EMG_Window_GetData(&hemg->windows[channel], temp_buffer);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Apply filtering */
    status = EMG_ApplyFilters(hemg, channel, temp_buffer, filtered_buffer, EMG_WINDOW_SIZE);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Extract features */
    status = EMG_ExtractFeatures(hemg, channel, filtered_buffer, EMG_WINDOW_SIZE);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Check if all channels are processed */
    static uint8_t channels_processed = 0;
    channels_processed++;
    
    if (channels_processed >= EMG_CHANNELS) {
        channels_processed = 0;
        
        /* Update feature timestamp */
        hemg->current_features.timestamp = HAL_GetTick() * 1000;
        hemg->current_features.is_valid = true;
        
        /* Normalize and quantize features */
        EMG_NormalizeFeatures(hemg);
        EMG_QuantizeFeatures(hemg);
        
        /* Update normalization parameters periodically */
        if (hemg->feature_count % EMG_NORMALIZATION_SAMPLES == 0) {
            EMG_UpdateNormalizationParams(hemg);
        }
        
        /* Call feature ready callback */
        if (hemg->feature_ready_callback != NULL) {
            hemg->feature_ready_callback(&hemg->current_features);
        }
        
        /* Update feature buffer */
        EMG_UpdateFeatureBuffer(hemg);
        
        hemg->feature_count++;
    }
    
    return EMG_OK;
}

/**
 * @brief Apply filtering to signal
 * @param hemg: EMG processing handle pointer
 * @param channel: Channel number
 * @param input: Input signal
 * @param output: Output signal
 * @param size: Signal size
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_ApplyFilters(EMG_ProcessingTypeDef *hemg, uint8_t channel, 
                                  float32_t *input, float32_t *output, uint16_t size)
{
    if (hemg == NULL || input == NULL || output == NULL || channel >= EMG_CHANNELS) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Apply bandpass filter */
    status = EMG_Filter_Process(&hemg->bandpass_filter[channel], input, output, size);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Apply notch filter */
    status = EMG_Filter_Process(&hemg->notch_filter[channel], output, output, size);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Apply comb filter for additional powerline interference removal */
    status = EMG_Filter_Process(&hemg->comb_filter[channel], output, output, size);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief Extract features from signal
 * @param hemg: EMG processing handle pointer
 * @param channel: Channel number
 * @param window_data: Window data
 * @param size: Window size
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_ExtractFeatures(EMG_ProcessingTypeDef *hemg, uint8_t channel, 
                                     float32_t *window_data, uint16_t size)
{
    if (hemg == NULL || window_data == NULL || channel >= EMG_CHANNELS) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Extract time domain features */
    status = EMG_ExtractTimeDomainFeatures(window_data, size,
                                          &hemg->current_features.mav[channel],
                                          &hemg->current_features.wl[channel],
                                          &hemg->current_features.zc[channel],
                                          &hemg->current_features.ssc[channel],
                                          &hemg->current_features.rms[channel]);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Extract frequency domain features */
    status = EMG_ExtractFrequencyDomainFeatures(window_data, size,
                                               &hemg->current_features.mean_freq[channel],
                                               &hemg->current_features.median_freq[channel],
                                               &hemg->current_features.total_power[channel],
                                               &hemg->current_features.freq_ratio[channel]);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief Extract time domain features
 * @param data: Signal data
 * @param size: Signal size
 * @param mav: Mean absolute value
 * @param wl: Waveform length
 * @param zc: Zero crossings
 * @param ssc: Slope sign changes
 * @param rms: Root mean square
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_ExtractTimeDomainFeatures(float32_t *data, uint16_t size, 
                                               float32_t *mav, float32_t *wl, 
                                               float32_t *zc, float32_t *ssc, float32_t *rms)
{
    if (data == NULL || mav == NULL || wl == NULL || zc == NULL || ssc == NULL || rms == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    float32_t sum_abs = 0.0f;
    float32_t sum_diff = 0.0f;
    float32_t sum_sq = 0.0f;
    uint16_t zero_count = 0;
    uint16_t slope_count = 0;
    
    /* Calculate features in single pass */
    for (uint16_t i = 0; i < size; i++) {
        /* MAV and RMS */
        sum_abs += fabsf(data[i]);
        sum_sq += data[i] * data[i];
        
        /* Waveform length */
        if (i > 0) {
            sum_diff += fabsf(data[i] - data[i-1]);
        }
        
        /* Zero crossings */
        if (i > 0 && ((data[i] > 0 && data[i-1] < 0) || (data[i] < 0 && data[i-1] > 0))) {
            zero_count++;
        }
        
        /* Slope sign changes */
        if (i > 1) {
            float32_t slope1 = data[i] - data[i-1];
            float32_t slope2 = data[i-1] - data[i-2];
            if ((slope1 > 0 && slope2 < 0) || (slope1 < 0 && slope2 > 0)) {
                slope_count++;
            }
        }
    }
    
    /* Calculate final features */
    *mav = sum_abs / size;
    *wl = sum_diff;
    *zc = (float32_t)zero_count;
    *ssc = (float32_t)slope_count;
    *rms = sqrtf(sum_sq / size);
    
    return EMG_OK;
}

/**
 * @brief Extract frequency domain features
 * @param data: Signal data
 * @param size: Signal size
 * @param mean_freq: Mean frequency
 * @param median_freq: Median frequency
 * @param total_power: Total power
 * @param freq_ratio: Frequency ratio
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_ExtractFrequencyDomainFeatures(float32_t *data, uint16_t size, 
                                                    float32_t *mean_freq, float32_t *median_freq, 
                                                    float32_t *total_power, float32_t *freq_ratio)
{
    if (data == NULL || mean_freq == NULL || median_freq == NULL || 
        total_power == NULL || freq_ratio == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Get power spectrum */
    EMG_StatusTypeDef status = EMG_FFT_GetPowerSpectrum(&emg_processing, data, 
                                                       emg_processing.magnitude_spectrum, size);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Calculate frequency features */
    *mean_freq = EMG_CalculateMeanFrequency(emg_processing.magnitude_spectrum, 
                                           DSP_FREQ_BINS, EMG_SAMPLING_RATE);
    *median_freq = EMG_CalculateMedianFrequency(emg_processing.magnitude_spectrum, 
                                               DSP_FREQ_BINS, EMG_SAMPLING_RATE);
    *total_power = EMG_CalculateTotalPower(emg_processing.magnitude_spectrum, DSP_FREQ_BINS);
    *freq_ratio = EMG_CalculateFrequencyRatio(emg_processing.magnitude_spectrum, DSP_FREQ_BINS, 
                                             20.0f, 200.0f, EMG_SAMPLING_RATE);
    
    return EMG_OK;
}

/**
 * @brief Normalize features
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_NormalizeFeatures(EMG_ProcessingTypeDef *hemg)
{
    if (hemg == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Normalize features using z-score normalization */
    uint16_t idx = 0;
    
    for (uint8_t ch = 0; ch < EMG_CHANNELS; ch++) {
        /* Time domain features */
        hemg->current_features.mav[ch] = (hemg->current_features.mav[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.wl[ch] = (hemg->current_features.wl[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.zc[ch] = (hemg->current_features.zc[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.ssc[ch] = (hemg->current_features.ssc[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.rms[ch] = (hemg->current_features.rms[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        
        /* Frequency domain features */
        hemg->current_features.mean_freq[ch] = (hemg->current_features.mean_freq[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.median_freq[ch] = (hemg->current_features.median_freq[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.total_power[ch] = (hemg->current_features.total_power[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
        hemg->current_features.freq_ratio[ch] = (hemg->current_features.freq_ratio[ch] - hemg->feature_mean[idx]) / hemg->feature_std[idx++];
    }
    
    return EMG_OK;
}

/**
 * @brief Quantize features for MAX78000
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef EMG_QuantizeFeatures(EMG_ProcessingTypeDef *hemg)
{
    if (hemg == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Quantize to 8-bit signed integers [-128, 127] */
    uint16_t idx = 0;
    
    for (uint8_t ch = 0; ch < EMG_CHANNELS; ch++) {
        /* Time domain features */
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.mav[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.wl[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.zc[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.ssc[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.rms[ch] * 127.0f));
        
        /* Frequency domain features */
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.mean_freq[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.median_freq[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.total_power[ch] * 127.0f));
        hemg->current_features.normalized_features[idx++] = (int8_t)fmaxf(-128.0f, fminf(127.0f, hemg->current_features.freq_ratio[ch] * 127.0f));
    }
    
    return EMG_OK;
}

/* Private functions ---------------------------------------------------------*/

/**
 * @brief Initialize filters
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef EMG_InitializeFilters(EMG_ProcessingTypeDef *hemg)
{
    EMG_StatusTypeDef status;
    
    for (uint8_t i = 0; i < EMG_CHANNELS; i++) {
        /* Initialize bandpass filter */
        status = EMG_Filter_Init(&hemg->bandpass_filter[i], FILTER_BANDPASS, 
                                DSP_BANDPASS_HIGH_FREQ, 0.707f);
        if (status != EMG_OK) {
            return status;
        }
        
        /* Initialize notch filter */
        status = EMG_Filter_Init(&hemg->notch_filter[i], FILTER_NOTCH, 
                                DSP_NOTCH_FREQ, 30.0f);
        if (status != EMG_OK) {
            return status;
        }
        
        /* Initialize comb filter */
        status = EMG_Filter_Init(&hemg->comb_filter[i], FILTER_COMB, 
                                DSP_NOTCH_FREQ, 1.0f);
        if (status != EMG_OK) {
            return status;
        }
    }
    
    return EMG_OK;
}

/**
 * @brief Initialize windows
 * @param hemg: EMG processing handle pointer
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef EMG_InitializeWindows(EMG_ProcessingTypeDef *hemg)
{
    EMG_StatusTypeDef status;
    
    for (uint8_t i = 0; i < EMG_CHANNELS; i++) {
        status = EMG_Window_Init(&hemg->windows[i], EMG_WINDOW_SIZE, EMG_OVERLAP);
        if (status != EMG_OK) {
            return status;
        }
    }
    
    return EMG_OK;
}

/**
 * @brief Update feature buffer
 * @param hemg: EMG processing handle pointer
 * @retval None
 */
static void EMG_UpdateFeatureBuffer(EMG_ProcessingTypeDef *hemg)
{
    /* Copy current features to previous */
    hemg->previous_features = hemg->current_features;
    
    /* Clear current features */
    memset(&hemg->current_features, 0, sizeof(hemg->current_features));
}

/* Remaining functions would be implemented similarly... */
/* This is a substantial implementation showing the core structure */