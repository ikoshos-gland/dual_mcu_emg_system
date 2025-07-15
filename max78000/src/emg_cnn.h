/**
 * @file emg_cnn.h
 * @brief MAX78000 CNN Implementation for EMG Classification Header
 * @author Dual-MCU EMG System
 * @date 2025
 */

#ifndef EMG_CNN_H
#define EMG_CNN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include <stdbool.h>

/* CNN Configuration */
#define EMG_CNN_INPUT_SIZE          128         // Input features (padded to 128)
#define EMG_CNN_FEATURE_SIZE        72          // Actual feature count (8 channels × 9 features)
#define EMG_CNN_OUTPUT_SIZE         8           // Number of output classes
#define EMG_CNN_LAYER_COUNT         6           // Number of CNN layers
#define EMG_CNN_WEIGHT_SIZE         8192        // Weight memory size (words)
#define EMG_CNN_BIAS_SIZE           256         // Bias memory size (words)

/* Generated Model Configuration */
#ifdef CNN_MODEL_AVAILABLE
#include "emg_cnn_generated.h"
#include "emg_weights.h"
#endif

/* EMG Gesture Classes */
typedef enum {
    EMG_CLASS_REST = 0,                         // Rest state
    EMG_CLASS_GRASP,                            // Grasp/close
    EMG_CLASS_RELEASE,                          // Release/open
    EMG_CLASS_ROTATE_CW,                        // Rotate clockwise
    EMG_CLASS_ROTATE_CCW,                       // Rotate counter-clockwise
    EMG_CLASS_FLEX,                             // Flex/bend
    EMG_CLASS_EXTEND,                           // Extend/straighten
    EMG_CLASS_POINT                             // Point gesture
} EMG_CNN_ClassTypeDef;

/* CNN Status */
typedef enum {
    EMG_CNN_OK = 0,                             // Success
    EMG_CNN_ERROR_INVALID_PARAM,                // Invalid parameter
    EMG_CNN_ERROR_NOT_READY,                    // CNN not ready
    EMG_CNN_ERROR_TIMEOUT,                      // Timeout error
    EMG_CNN_ERROR_INFERENCE,                    // Inference error
    EMG_CNN_ERROR_MEMORY                        // Memory error
} EMG_CNN_StatusTypeDef;

/* CNN Result Structure */
typedef struct {
    uint8_t predicted_class;                    // Predicted class index
    uint16_t confidence;                        // Confidence score (0-1000)
    uint16_t class_scores[EMG_CNN_OUTPUT_SIZE]; // Individual class scores
    uint32_t inference_time;                    // Inference time in microseconds
    uint32_t timestamp;                         // Timestamp of inference
} EMG_CNN_ResultTypeDef;

/* CNN Status Structure */
typedef struct {
    bool is_enabled;                            // CNN enabled flag
    bool is_ready;                              // CNN ready flag
    bool is_complete;                           // Inference complete flag
    uint32_t layer_count;                       // Current layer count
    uint32_t memory_count;                      // Memory usage count
} EMG_CNN_StatusTypeDef;

/* CNN Metrics Structure */
typedef struct {
    uint32_t last_inference_time;               // Last inference time (µs)
    uint32_t total_inferences;                  // Total inference count
    uint32_t average_inference_time;            // Average inference time (µs)
    uint32_t peak_inference_time;               // Peak inference time (µs)
    uint8_t processor_utilization;              // Processor utilization (%)
} EMG_CNN_MetricsTypeDef;

/* Callback Function Type */
typedef void (*EMG_CNN_CallbackTypeDef)(void);

/* Function Prototypes */

/* Initialization and Configuration */
EMG_CNN_StatusTypeDef EMG_CNN_Init(EMG_CNN_CallbackTypeDef callback);
EMG_CNN_StatusTypeDef EMG_CNN_DeInit(void);
EMG_CNN_StatusTypeDef EMG_CNN_Reset(void);

/* Data Loading and Inference */
EMG_CNN_StatusTypeDef EMG_CNN_LoadFeatures(const int8_t *features);
EMG_CNN_StatusTypeDef EMG_CNN_StartInference(void);
EMG_CNN_StatusTypeDef EMG_CNN_GetResults(EMG_CNN_ResultTypeDef *results);
EMG_CNN_StatusTypeDef EMG_CNN_WaitComplete(uint32_t timeout_ms);

/* Status and Monitoring */
bool EMG_CNN_IsReady(void);
bool EMG_CNN_IsComplete(void);
EMG_CNN_StatusTypeDef EMG_CNN_GetStatus(EMG_CNN_StatusTypeDef *status);
EMG_CNN_StatusTypeDef EMG_CNN_GetMetrics(EMG_CNN_MetricsTypeDef *metrics);

/* Utility Functions */
const char* EMG_CNN_GetClassName(uint8_t class_id);
uint16_t EMG_CNN_GetConfidenceThreshold(void);
void EMG_CNN_SetConfidenceThreshold(uint16_t threshold);

#ifdef __cplusplus
}
#endif

#endif /* EMG_CNN_H */