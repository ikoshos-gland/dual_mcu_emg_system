/**
 * @file emg_processing.h
 * @brief EMG Signal Processing and Feature Extraction Header
 * @author Dual-MCU EMG System
 * @date 2025
 */

#ifndef EMG_PROCESSING_H
#define EMG_PROCESSING_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "ads1299_driver.h"
#include "arm_math.h"
#include <stdint.h>
#include <stdbool.h>

/* DSP Configuration */
#define DSP_FILTER_ORDER            4               // IIR filter order
#define DSP_FFT_SIZE                512             // FFT size
#define DSP_FREQ_BINS               256             // Frequency bins
#define DSP_OVERLAP_FACTOR          0.5f            // 50% overlap
#define DSP_WINDOW_HANN             1               // Hann window
#define DSP_WINDOW_HAMMING          2               // Hamming window

/* Feature Extraction Configuration */
#define FEATURE_TIME_DOMAIN         5               // Time domain features per channel
#define FEATURE_FREQ_DOMAIN         4               // Frequency domain features per channel
#define FEATURE_TOTAL_PER_CHANNEL   (FEATURE_TIME_DOMAIN + FEATURE_FREQ_DOMAIN)
#define FEATURE_VECTOR_SIZE         (FEATURE_TOTAL_PER_CHANNEL * EMG_CHANNELS)

/* Filter Types */
typedef enum {
    FILTER_BANDPASS = 0,
    FILTER_HIGHPASS,
    FILTER_LOWPASS,
    FILTER_NOTCH,
    FILTER_COMB
} FilterTypeTypeDef;

/* Filter Structure */
typedef struct {
    float32_t coeffs_b[DSP_FILTER_ORDER + 1];      // Numerator coefficients
    float32_t coeffs_a[DSP_FILTER_ORDER + 1];      // Denominator coefficients
    float32_t state[DSP_FILTER_ORDER * 2];         // Filter state
    arm_biquad_casd_df1_inst_f32 instance;         // ARM DSP filter instance
    FilterTypeTypeDef type;                         // Filter type
    float32_t cutoff_freq;                          // Cutoff frequency
    float32_t Q_factor;                             // Q factor for bandpass/notch
    bool is_initialized;                            // Initialization flag
} EMG_FilterTypeDef;

/* Processing Window Structure */
typedef struct {
    float32_t *data;                                // Window data
    uint16_t size;                                  // Window size
    uint16_t overlap;                               // Overlap samples
    uint16_t current_index;                         // Current sample index
    bool is_full;                                   // Window full flag
} EMG_WindowTypeDef;

/* Feature Vector Structure */
typedef struct {
    // Time domain features (per channel)
    float32_t mav[EMG_CHANNELS];                    // Mean Absolute Value
    float32_t wl[EMG_CHANNELS];                     // Waveform Length
    float32_t zc[EMG_CHANNELS];                     // Zero Crossings
    float32_t ssc[EMG_CHANNELS];                    // Slope Sign Changes
    float32_t rms[EMG_CHANNELS];                    // Root Mean Square
    
    // Frequency domain features (per channel)
    float32_t mean_freq[EMG_CHANNELS];              // Mean Frequency
    float32_t median_freq[EMG_CHANNELS];            // Median Frequency
    float32_t total_power[EMG_CHANNELS];            // Total Power
    float32_t freq_ratio[EMG_CHANNELS];             // Frequency Ratio
    
    // Normalized feature vector for MAX78000
    int8_t normalized_features[FEATURE_VECTOR_SIZE]; // Quantized features
    
    uint32_t timestamp;                             // Feature timestamp
    bool is_valid;                                  // Feature validity flag
} EMG_FeatureVectorTypeDef;

/* Processing Context Structure */
typedef struct {
    // Input data buffer
    float32_t input_buffer[EMG_CHANNELS][EMG_BUFFER_SIZE];
    uint16_t buffer_head;
    uint16_t buffer_tail;
    
    // Processing windows
    EMG_WindowTypeDef windows[EMG_CHANNELS];
    
    // Filters
    EMG_FilterTypeDef bandpass_filter[EMG_CHANNELS];
    EMG_FilterTypeDef notch_filter[EMG_CHANNELS];
    EMG_FilterTypeDef comb_filter[EMG_CHANNELS];
    
    // FFT instances
    arm_rfft_fast_instance_f32 fft_instance;
    float32_t fft_buffer[DSP_FFT_SIZE];
    float32_t fft_output[DSP_FFT_SIZE];
    float32_t magnitude_spectrum[DSP_FREQ_BINS];
    
    // Feature extraction
    EMG_FeatureVectorTypeDef current_features;
    EMG_FeatureVectorTypeDef previous_features;
    
    // Normalization parameters
    float32_t feature_min[FEATURE_VECTOR_SIZE];
    float32_t feature_max[FEATURE_VECTOR_SIZE];
    float32_t feature_mean[FEATURE_VECTOR_SIZE];
    float32_t feature_std[FEATURE_VECTOR_SIZE];
    
    // Processing state
    bool is_initialized;
    bool is_running;
    uint32_t sample_count;
    uint32_t feature_count;
    
    // Callback functions
    void (*feature_ready_callback)(EMG_FeatureVectorTypeDef *features);
    void (*error_callback)(EMG_StatusTypeDef error);
} EMG_ProcessingTypeDef;

/* Function Prototypes */

/* Initialization and Configuration */
EMG_StatusTypeDef EMG_Processing_Init(EMG_ProcessingTypeDef *hemg);
EMG_StatusTypeDef EMG_Processing_DeInit(EMG_ProcessingTypeDef *hemg);
EMG_StatusTypeDef EMG_Processing_Start(EMG_ProcessingTypeDef *hemg);
EMG_StatusTypeDef EMG_Processing_Stop(EMG_ProcessingTypeDef *hemg);

/* Filter Functions */
EMG_StatusTypeDef EMG_Filter_Init(EMG_FilterTypeDef *filter, FilterTypeTypeDef type, 
                                 float32_t cutoff_freq, float32_t Q_factor);
EMG_StatusTypeDef EMG_Filter_Process(EMG_FilterTypeDef *filter, float32_t *input, 
                                    float32_t *output, uint16_t size);
EMG_StatusTypeDef EMG_Filter_Reset(EMG_FilterTypeDef *filter);

/* Signal Processing Functions */
EMG_StatusTypeDef EMG_ProcessSample(EMG_ProcessingTypeDef *hemg, ADS1299_DataTypeDef *data);
EMG_StatusTypeDef EMG_ProcessWindow(EMG_ProcessingTypeDef *hemg, uint8_t channel);
EMG_StatusTypeDef EMG_ApplyFilters(EMG_ProcessingTypeDef *hemg, uint8_t channel, 
                                  float32_t *input, float32_t *output, uint16_t size);

/* Feature Extraction Functions */
EMG_StatusTypeDef EMG_ExtractFeatures(EMG_ProcessingTypeDef *hemg, uint8_t channel, 
                                     float32_t *window_data, uint16_t size);
EMG_StatusTypeDef EMG_ExtractTimeDomainFeatures(float32_t *data, uint16_t size, 
                                               float32_t *mav, float32_t *wl, 
                                               float32_t *zc, float32_t *ssc, float32_t *rms);
EMG_StatusTypeDef EMG_ExtractFrequencyDomainFeatures(float32_t *data, uint16_t size, 
                                                    float32_t *mean_freq, float32_t *median_freq, 
                                                    float32_t *total_power, float32_t *freq_ratio);

/* Utility Functions */
EMG_StatusTypeDef EMG_NormalizeFeatures(EMG_ProcessingTypeDef *hemg);
EMG_StatusTypeDef EMG_QuantizeFeatures(EMG_ProcessingTypeDef *hemg);
EMG_StatusTypeDef EMG_UpdateNormalizationParams(EMG_ProcessingTypeDef *hemg);

/* Window Functions */
EMG_StatusTypeDef EMG_Window_Init(EMG_WindowTypeDef *window, uint16_t size, uint16_t overlap);
EMG_StatusTypeDef EMG_Window_AddSample(EMG_WindowTypeDef *window, float32_t sample);
bool EMG_Window_IsReady(EMG_WindowTypeDef *window);
EMG_StatusTypeDef EMG_Window_GetData(EMG_WindowTypeDef *window, float32_t *output);

/* FFT Functions */
EMG_StatusTypeDef EMG_FFT_Init(EMG_ProcessingTypeDef *hemg);
EMG_StatusTypeDef EMG_FFT_Process(EMG_ProcessingTypeDef *hemg, float32_t *input, 
                                 float32_t *magnitude, uint16_t size);
EMG_StatusTypeDef EMG_FFT_GetPowerSpectrum(EMG_ProcessingTypeDef *hemg, float32_t *input, 
                                          float32_t *power_spectrum, uint16_t size);

/* Callback Functions */
void EMG_DataReadyCallback(ADS1299_DataTypeDef *data);
void EMG_FeatureReadyCallback(EMG_FeatureVectorTypeDef *features);
void EMG_ErrorCallback(EMG_StatusTypeDef error);

/* Statistics Functions */
float32_t EMG_CalculateMean(float32_t *data, uint16_t size);
float32_t EMG_CalculateVariance(float32_t *data, uint16_t size);
float32_t EMG_CalculateStdDev(float32_t *data, uint16_t size);
float32_t EMG_CalculateRMS(float32_t *data, uint16_t size);
uint16_t EMG_CountZeroCrossings(float32_t *data, uint16_t size);
uint16_t EMG_CountSlopeSignChanges(float32_t *data, uint16_t size);

/* Frequency Analysis Functions */
float32_t EMG_CalculateMeanFrequency(float32_t *power_spectrum, uint16_t size, float32_t fs);
float32_t EMG_CalculateMedianFrequency(float32_t *power_spectrum, uint16_t size, float32_t fs);
float32_t EMG_CalculateTotalPower(float32_t *power_spectrum, uint16_t size);
float32_t EMG_CalculateFrequencyRatio(float32_t *power_spectrum, uint16_t size, 
                                     float32_t f1, float32_t f2, float32_t fs);

/* External Variables */
extern EMG_ProcessingTypeDef emg_processing;

#ifdef __cplusplus
}
#endif

#endif /* EMG_PROCESSING_H */