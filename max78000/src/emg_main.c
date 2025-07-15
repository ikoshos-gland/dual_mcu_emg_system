/**
 * @file emg_main.c
 * @brief MAX78000 Main Application for EMG Classification
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_device.h"
#include "mxc_delay.h"
#include "mxc_sys.h"
#include "board.h"
#include "led.h"
#include "pb.h"
#include "tmr.h"
#include "spi.h"
#include "emg_cnn.h"
#include "emg_communication.h"

/* Private defines -----------------------------------------------------------*/
#define APP_VERSION                 "1.0.0"
#define CONFIDENCE_THRESHOLD        700         // 70% confidence threshold
#define INFERENCE_TIMEOUT_MS        100         // 100ms timeout for inference
#define COMMUNICATION_TIMEOUT_MS    50          // 50ms timeout for communication

/* Private variables ---------------------------------------------------------*/
static volatile bool feature_data_ready = false;
static volatile bool inference_complete = false;
static int8_t feature_buffer[EMG_CNN_FEATURE_SIZE];
static EMG_CNN_ResultTypeDef inference_result;
static uint32_t inference_count = 0;
static uint32_t classification_count = 0;

/* Private function prototypes -----------------------------------------------*/
static void System_Init(void);
static void Hardware_Init(void);
static void Communication_Init(void);
static void Process_FeatureData(void);
static void Process_InferenceResult(void);
static void Update_Status_LED(uint8_t class_id);
static void Print_Results(const EMG_CNN_ResultTypeDef *result);

/* Callback functions */
static void Feature_Data_Callback(const int8_t *features, uint16_t size);
static void Inference_Complete_Callback(void);
static void Communication_Error_Callback(uint8_t error_code);

/* Class names for display */
static const char* class_names[EMG_CNN_OUTPUT_SIZE] = {
    "Rest",
    "Grasp",
    "Release", 
    "Rotate CW",
    "Rotate CCW",
    "Flex",
    "Extend",
    "Point"
};

/* Main function -------------------------------------------------------------*/
int main(void)
{
    printf("\n\n***** MAX78000 EMG Classification System *****\n");
    printf("Version: %s\n", APP_VERSION);
    printf("Build Date: %s %s\n\n", __DATE__, __TIME__);
    
    /* Initialize system */
    System_Init();
    
    /* Initialize hardware */
    Hardware_Init();
    
    /* Initialize communication */
    Communication_Init();
    
    /* Initialize CNN */
    if (EMG_CNN_Init(Inference_Complete_Callback) != EMG_CNN_OK) {
        printf("ERROR: CNN initialization failed\n");
        while(1);
    }
    
    printf("System initialized successfully\n");
    printf("Waiting for feature data from STM32...\n\n");
    
    /* Main application loop */
    while (1) {
        /* Check for new feature data */
        if (feature_data_ready) {
            Process_FeatureData();
            feature_data_ready = false;
        }
        
        /* Check for inference completion */
        if (inference_complete) {
            Process_InferenceResult();
            inference_complete = false;
        }
        
        /* Handle communication */
        EMG_Communication_Process();
        
        /* Small delay to prevent busy waiting */
        MXC_Delay(1);
    }
    
    return 0;
}

/* Private functions ---------------------------------------------------------*/

/**
 * @brief System initialization
 * @retval None
 */
static void System_Init(void)
{
    /* Initialize system clock */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();
    
    /* Enable instruction cache */
    MXC_ICC_Enable(MXC_ICC0);
    
    /* Initialize console UART */
    Console_Init();
    
    /* Initialize timer for performance measurement */
    MXC_TMR_Init(MXC_TMR0, MXC_TMR_PRES_1, false);
}

/**
 * @brief Hardware initialization
 * @retval None
 */
static void Hardware_Init(void)
{
    /* Initialize LEDs */
    LED_Init();
    LED_Off(0);
    LED_Off(1);
    LED_Off(2);
    
    /* Initialize push buttons */
    PB_Init();
    
    /* Set initial LED pattern */
    LED_On(0); // Power/ready indicator
}

/**
 * @brief Communication initialization
 * @retval None
 */
static void Communication_Init(void)
{
    EMG_Communication_Config config = {
        .spi_instance = MXC_SPI1,
        .cs_port = MXC_GPIO0,
        .cs_pin = MXC_GPIO_PIN_4,
        .irq_port = MXC_GPIO0,
        .irq_pin = MXC_GPIO_PIN_5,
        .baudrate = 10000000, // 10MHz
        .feature_callback = Feature_Data_Callback,
        .error_callback = Communication_Error_Callback
    };
    
    if (EMG_Communication_Init(&config) != EMG_COMM_OK) {
        printf("ERROR: Communication initialization failed\n");
        while(1);
    }
}

/**
 * @brief Process received feature data
 * @retval None
 */
static void Process_FeatureData(void)
{
    EMG_CNN_StatusTypeDef status;
    
    /* Check if CNN is ready */
    if (!EMG_CNN_IsReady()) {
        printf("WARNING: CNN not ready for new data\n");
        return;
    }
    
    /* Load features into CNN */
    status = EMG_CNN_LoadFeatures(feature_buffer);
    if (status != EMG_CNN_OK) {
        printf("ERROR: Failed to load features (status: %d)\n", status);
        return;
    }
    
    /* Start inference */
    status = EMG_CNN_StartInference();
    if (status != EMG_CNN_OK) {
        printf("ERROR: Failed to start inference (status: %d)\n", status);
        return;
    }
    
    inference_count++;
    LED_Toggle(1); // Activity indicator
}

/**
 * @brief Process inference result
 * @retval None
 */
static void Process_InferenceResult(void)
{
    EMG_CNN_StatusTypeDef status;
    
    /* Get inference results */
    status = EMG_CNN_GetResults(&inference_result);
    if (status != EMG_CNN_OK) {
        printf("ERROR: Failed to get results (status: %d)\n", status);
        return;
    }
    
    /* Check confidence threshold */
    if (inference_result.confidence >= CONFIDENCE_THRESHOLD) {
        classification_count++;
        
        /* Update status LED */
        Update_Status_LED(inference_result.predicted_class);
        
        /* Print results */
        Print_Results(&inference_result);
        
        /* Send classification result back to STM32 */
        EMG_Communication_SendClassification(&inference_result);
    } else {
        printf("Low confidence: %d%% (threshold: %d%%)\n", 
               inference_result.confidence / 10, CONFIDENCE_THRESHOLD / 10);
    }
}

/**
 * @brief Update status LED based on classification
 * @param class_id: Predicted class ID
 * @retval None
 */
static void Update_Status_LED(uint8_t class_id)
{
    /* Turn off all LEDs */
    LED_Off(0);
    LED_Off(1);
    LED_Off(2);
    
    /* Light LED based on class */
    switch (class_id) {
        case EMG_CLASS_REST:
            LED_On(0); // Green for rest
            break;
        case EMG_CLASS_GRASP:
        case EMG_CLASS_RELEASE:
            LED_On(1); // Yellow for grasp/release
            break;
        case EMG_CLASS_ROTATE_CW:
        case EMG_CLASS_ROTATE_CCW:
            LED_On(2); // Red for rotation
            break;
        default:
            LED_On(0); // Default to green
            break;
    }
}

/**
 * @brief Print inference results
 * @param result: Pointer to result structure
 * @retval None
 */
static void Print_Results(const EMG_CNN_ResultTypeDef *result)
{
    printf("Inference #%lu: %s (%.1f%%) - %lu μs\n",
           classification_count,
           class_names[result->predicted_class],
           result->confidence / 10.0f,
           result->inference_time);
    
    /* Print detailed class scores */
    printf("  Scores: ");
    for (int i = 0; i < EMG_CNN_OUTPUT_SIZE; i++) {
        printf("%s:%.1f%% ", class_names[i], result->class_scores[i] / 10.0f);
    }
    printf("\n\n");
}

/* Callback functions --------------------------------------------------------*/

/**
 * @brief Feature data received callback
 * @param features: Feature data array
 * @param size: Data size
 * @retval None
 */
static void Feature_Data_Callback(const int8_t *features, uint16_t size)
{
    if (size == EMG_CNN_FEATURE_SIZE) {
        memcpy(feature_buffer, features, size);
        feature_data_ready = true;
    } else {
        printf("ERROR: Invalid feature data size: %d (expected: %d)\n", 
               size, EMG_CNN_FEATURE_SIZE);
    }
}

/**
 * @brief Inference completion callback
 * @retval None
 */
static void Inference_Complete_Callback(void)
{
    inference_complete = true;
}

/**
 * @brief Communication error callback
 * @param error_code: Error code
 * @retval None
 */
static void Communication_Error_Callback(uint8_t error_code)
{
    printf("Communication error: %d\n", error_code);
    
    /* Reset communication if needed */
    if (error_code == EMG_COMM_ERROR_TIMEOUT) {
        EMG_Communication_Reset();
    }
}

/* System status functions ---------------------------------------------------*/

/**
 * @brief Print system status
 * @retval None
 */
void Print_System_Status(void)
{
    EMG_CNN_MetricsTypeDef metrics;
    EMG_CNN_StatusTypeDef cnn_status;
    
    /* Get CNN metrics */
    if (EMG_CNN_GetMetrics(&metrics) == EMG_CNN_OK) {
        printf("CNN Metrics:\n");
        printf("  Total Inferences: %lu\n", metrics.total_inferences);
        printf("  Average Time: %lu μs\n", metrics.average_inference_time);
        printf("  Peak Time: %lu μs\n", metrics.peak_inference_time);
        printf("  Processor Utilization: %d%%\n", metrics.processor_utilization);
    }
    
    /* Get CNN status */
    if (EMG_CNN_GetStatus(&cnn_status) == EMG_CNN_OK) {
        printf("CNN Status:\n");
        printf("  Enabled: %s\n", cnn_status.is_enabled ? "Yes" : "No");
        printf("  Ready: %s\n", cnn_status.is_ready ? "Yes" : "No");
        printf("  Layer Count: %lu\n", cnn_status.layer_count);
        printf("  Memory Count: %lu\n", cnn_status.memory_count);
    }
    
    printf("System Statistics:\n");
    printf("  Total Inferences: %lu\n", inference_count);
    printf("  Valid Classifications: %lu\n", classification_count);
    printf("  Success Rate: %.1f%%\n", 
           inference_count > 0 ? (classification_count * 100.0f / inference_count) : 0.0f);
}

/* Push button handler -------------------------------------------------------*/

/**
 * @brief Push button interrupt handler
 * @retval None
 */
void PB_Handler(void)
{
    if (PB_Get(0)) {
        Print_System_Status();
    }
}

/* Error handler -------------------------------------------------------------*/

/**
 * @brief Error handler
 * @retval None
 */
void Error_Handler(void)
{
    printf("*** SYSTEM ERROR ***\n");
    
    /* Turn on error LED */
    LED_On(2);
    
    /* Infinite loop */
    while (1) {
        LED_Toggle(2);
        MXC_Delay(500);
    }
}