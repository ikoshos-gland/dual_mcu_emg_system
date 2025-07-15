/**
 * @file emg_cnn.c
 * @brief MAX78000 CNN Implementation for EMG Classification
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_sys.h"
#include "bbfc_regs.h"
#include "fcr_regs.h"
#include "icc.h"
#include "mxc_device.h"
#include "mxc_delay.h"
#include "nvic_table.h"
#include "board.h"
#include "tmr.h"
#include "dma.h"
#include "led.h"
#include "pb.h"
#include "emg_cnn.h"
#include "emg_weights.h"

/* Private defines -----------------------------------------------------------*/
#define CNN_TIMEOUT_MS          1000
#define CNN_QUADRANT_SIZE       4
#define CNN_NUM_QUADRANTS       4
#define CNN_PROCESSORS_PER_QUAD 16
#define CNN_TOTAL_PROCESSORS    (CNN_NUM_QUADRANTS * CNN_PROCESSORS_PER_QUAD)

/* Private variables ---------------------------------------------------------*/
static uint32_t cnn_time;
static volatile bool cnn_complete = false;
static EMG_CNN_CallbackTypeDef inference_callback = NULL;

/* Private function prototypes -----------------------------------------------*/
static void CNN_Init_Processors(void);
static void CNN_Load_Weights(void);
static void CNN_Configure_Input(void);
static void CNN_Wait_Complete(void);
static void CNN_Interrupt_Handler(void);

/* CNN Memory Map */
#define CNN_CTRL_BASE           0x50100000
#define CNN_SRAM_BASE           0x50400000
#define CNN_BIAS_BASE           0x50180000
#define CNN_TRAM_BASE           0x50800000

/* CNN Control Registers */
#define CNN_CTRL_REG            (*(volatile uint32_t *)(CNN_CTRL_BASE + 0x00))
#define CNN_SRAM_REG            (*(volatile uint32_t *)(CNN_CTRL_BASE + 0x04))
#define CNN_LCNT_REG            (*(volatile uint32_t *)(CNN_CTRL_BASE + 0x08))
#define CNN_MCNT_REG            (*(volatile uint32_t *)(CNN_CTRL_BASE + 0x0C))
#define CNN_TRAM_REG            (*(volatile uint32_t *)(CNN_CTRL_BASE + 0x10))
#define CNN_IFRM_REG            (*(volatile uint32_t *)(CNN_CTRL_BASE + 0x14))

/* CNN Status Bits */
#define CNN_STAT_COMPLETE       (1 << 12)
#define CNN_STAT_READY          (1 << 8)
#define CNN_CTRL_ENABLE         (1 << 0)

/* Public functions ----------------------------------------------------------*/

/**
 * @brief Initialize CNN accelerator
 * @param callback: Callback function for inference completion
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_Init(EMG_CNN_CallbackTypeDef callback)
{
    inference_callback = callback;
    
    /* Enable CNN clock */
    MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN);
    
    /* Reset CNN */
    MXC_SYS_Reset_Periph(MXC_SYS_RESET_CNN);
    
    /* Initialize CNN processors */
    CNN_Init_Processors();
    
    /* Load neural network weights */
    CNN_Load_Weights();
    
    /* Configure input layer */
    CNN_Configure_Input();
    
    /* Enable CNN interrupt */
    NVIC_SetVector(CNN_IRQn, (uint32_t)CNN_Interrupt_Handler);
    NVIC_EnableIRQ(CNN_IRQn);
    
    /* Enable CNN accelerator */
    CNN_CTRL_REG |= CNN_CTRL_ENABLE;
    
    return EMG_CNN_OK;
}

/**
 * @brief De-initialize CNN accelerator
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_DeInit(void)
{
    /* Disable CNN interrupt */
    NVIC_DisableIRQ(CNN_IRQn);
    
    /* Disable CNN accelerator */
    CNN_CTRL_REG &= ~CNN_CTRL_ENABLE;
    
    /* Reset CNN */
    MXC_SYS_Reset_Periph(MXC_SYS_RESET_CNN);
    
    /* Disable CNN clock */
    MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);
    
    inference_callback = NULL;
    
    return EMG_CNN_OK;
}

/**
 * @brief Load feature data into CNN
 * @param features: Feature data array (72 bytes)
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_LoadFeatures(const int8_t *features)
{
    if (features == NULL) {
        return EMG_CNN_ERROR_INVALID_PARAM;
    }
    
    /* Check if CNN is ready */
    if (!(CNN_CTRL_REG & CNN_STAT_READY)) {
        return EMG_CNN_ERROR_NOT_READY;
    }
    
    /* Load features into CNN SRAM */
    volatile uint32_t *sram_addr = (volatile uint32_t *)CNN_SRAM_BASE;
    
    /* Load 72 features (8 channels × 9 features each) */
    for (int i = 0; i < EMG_CNN_FEATURE_SIZE; i += 4) {
        uint32_t packed_data = 0;
        
        /* Pack 4 int8_t features into one uint32_t */
        packed_data |= (uint32_t)(features[i + 0] & 0xFF) << 0;
        packed_data |= (uint32_t)(features[i + 1] & 0xFF) << 8;
        packed_data |= (uint32_t)(features[i + 2] & 0xFF) << 16;
        packed_data |= (uint32_t)(features[i + 3] & 0xFF) << 24;
        
        *sram_addr++ = packed_data;
    }
    
    /* Zero-pad remaining input if needed */
    for (int i = EMG_CNN_FEATURE_SIZE; i < EMG_CNN_INPUT_SIZE; i += 4) {
        *sram_addr++ = 0;
    }
    
    return EMG_CNN_OK;
}

/**
 * @brief Start CNN inference
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_StartInference(void)
{
    /* Check if CNN is ready */
    if (!(CNN_CTRL_REG & CNN_STAT_READY)) {
        return EMG_CNN_ERROR_NOT_READY;
    }
    
    /* Clear completion flag */
    cnn_complete = false;
    
    /* Start timer for performance measurement */
    MXC_TMR_SW_Start(MXC_TMR0);
    
    /* Start CNN inference */
    CNN_CTRL_REG |= (1 << 1); // Start bit
    
    return EMG_CNN_OK;
}

/**
 * @brief Get inference results
 * @param results: Pointer to results structure
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_GetResults(EMG_CNN_ResultTypeDef *results)
{
    if (results == NULL) {
        return EMG_CNN_ERROR_INVALID_PARAM;
    }
    
    /* Check if inference is complete */
    if (!cnn_complete) {
        return EMG_CNN_ERROR_NOT_READY;
    }
    
    /* Read results from CNN TRAM */
    volatile uint32_t *tram_addr = (volatile uint32_t *)CNN_TRAM_BASE;
    int32_t raw_outputs[EMG_CNN_OUTPUT_SIZE];
    
    /* Read raw outputs */
    for (int i = 0; i < EMG_CNN_OUTPUT_SIZE; i++) {
        raw_outputs[i] = (int32_t)*tram_addr++;
    }
    
    /* Convert to probabilities and find best class */
    results->inference_time = cnn_time;
    results->predicted_class = 0;
    int32_t max_output = raw_outputs[0];
    
    for (int i = 0; i < EMG_CNN_OUTPUT_SIZE; i++) {
        /* Convert to probability (0-1000) */
        results->class_scores[i] = (uint16_t)((raw_outputs[i] + 128) * 1000 / 256);
        
        /* Find maximum */
        if (raw_outputs[i] > max_output) {
            max_output = raw_outputs[i];
            results->predicted_class = i;
        }
    }
    
    /* Calculate confidence */
    results->confidence = results->class_scores[results->predicted_class];
    
    /* Set timestamp */
    results->timestamp = MXC_TMR_GetTime();
    
    return EMG_CNN_OK;
}

/**
 * @brief Wait for inference completion (blocking)
 * @param timeout_ms: Timeout in milliseconds
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_WaitComplete(uint32_t timeout_ms)
{
    uint32_t start_time = MXC_TMR_GetTime();
    
    while (!cnn_complete) {
        if ((MXC_TMR_GetTime() - start_time) > timeout_ms) {
            return EMG_CNN_ERROR_TIMEOUT;
        }
        
        /* Small delay to prevent busy waiting */
        MXC_Delay(1);
    }
    
    return EMG_CNN_OK;
}

/**
 * @brief Check if CNN is ready for new inference
 * @retval true if ready
 */
bool EMG_CNN_IsReady(void)
{
    return (CNN_CTRL_REG & CNN_STAT_READY) != 0;
}

/**
 * @brief Check if inference is complete
 * @retval true if complete
 */
bool EMG_CNN_IsComplete(void)
{
    return cnn_complete;
}

/**
 * @brief Get CNN processor status
 * @param status: Pointer to status structure
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_GetStatus(EMG_CNN_StatusTypeDef *status)
{
    if (status == NULL) {
        return EMG_CNN_ERROR_INVALID_PARAM;
    }
    
    uint32_t ctrl_reg = CNN_CTRL_REG;
    
    status->is_enabled = (ctrl_reg & CNN_CTRL_ENABLE) != 0;
    status->is_ready = (ctrl_reg & CNN_STAT_READY) != 0;
    status->is_complete = (ctrl_reg & CNN_STAT_COMPLETE) != 0;
    status->layer_count = CNN_LCNT_REG;
    status->memory_count = CNN_MCNT_REG;
    
    return EMG_CNN_OK;
}

/* Private functions ---------------------------------------------------------*/

/**
 * @brief Initialize CNN processors
 * @retval None
 */
static void CNN_Init_Processors(void)
{
    /* Initialize all 64 processors */
    for (int quad = 0; quad < CNN_NUM_QUADRANTS; quad++) {
        for (int proc = 0; proc < CNN_PROCESSORS_PER_QUAD; proc++) {
            /* Configure processor registers */
            volatile uint32_t *proc_base = (volatile uint32_t *)(CNN_SRAM_BASE + 
                                                                (quad * 0x8000) + 
                                                                (proc * 0x400));
            
            /* Initialize processor control registers */
            proc_base[0] = 0x00000000; // Control register
            proc_base[1] = 0x00000000; // Status register
            proc_base[2] = 0x00000000; // Configuration register
        }
    }
}

/**
 * @brief Load CNN weights
 * @retval None
 */
static void CNN_Load_Weights(void)
{
    /* Load weights from generated header file */
    volatile uint32_t *weight_addr = (volatile uint32_t *)CNN_BIAS_BASE;
    
    /* Load kernel weights */
    for (int i = 0; i < EMG_CNN_WEIGHT_SIZE; i++) {
        *weight_addr++ = emg_cnn_weights[i];
    }
    
    /* Load bias values */
    volatile uint32_t *bias_addr = (volatile uint32_t *)(CNN_BIAS_BASE + 0x4000);
    for (int i = 0; i < EMG_CNN_BIAS_SIZE; i++) {
        *bias_addr++ = emg_cnn_bias[i];
    }
}

/**
 * @brief Configure CNN input layer
 * @retval None
 */
static void CNN_Configure_Input(void)
{
    /* Configure input frame register */
    CNN_IFRM_REG = 0x00000001; // Single frame input
    
    /* Configure SRAM for input data */
    CNN_SRAM_REG = 0x00000000; // Start from beginning of SRAM
    
    /* Configure layer count */
    CNN_LCNT_REG = EMG_CNN_LAYER_COUNT;
    
    /* Configure memory count */
    CNN_MCNT_REG = EMG_CNN_INPUT_SIZE / 4; // 32-bit words
}

/**
 * @brief CNN interrupt handler
 * @retval None
 */
static void CNN_Interrupt_Handler(void)
{
    /* Check if CNN is complete */
    if (CNN_CTRL_REG & CNN_STAT_COMPLETE) {
        /* Stop timer */
        cnn_time = MXC_TMR_SW_Stop(MXC_TMR0);
        
        /* Set completion flag */
        cnn_complete = true;
        
        /* Clear interrupt */
        CNN_CTRL_REG |= CNN_STAT_COMPLETE;
        
        /* Call callback if registered */
        if (inference_callback != NULL) {
            inference_callback();
        }
    }
}

/**
 * @brief Software reset CNN
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_Reset(void)
{
    /* Disable CNN */
    CNN_CTRL_REG &= ~CNN_CTRL_ENABLE;
    
    /* Reset CNN */
    MXC_SYS_Reset_Periph(MXC_SYS_RESET_CNN);
    
    /* Small delay */
    MXC_Delay(10);
    
    /* Re-initialize */
    CNN_Init_Processors();
    CNN_Load_Weights();
    CNN_Configure_Input();
    
    /* Enable CNN */
    CNN_CTRL_REG |= CNN_CTRL_ENABLE;
    
    return EMG_CNN_OK;
}

/**
 * @brief Get CNN performance metrics
 * @param metrics: Pointer to metrics structure
 * @retval EMG_CNN_StatusTypeDef
 */
EMG_CNN_StatusTypeDef EMG_CNN_GetMetrics(EMG_CNN_MetricsTypeDef *metrics)
{
    if (metrics == NULL) {
        return EMG_CNN_ERROR_INVALID_PARAM;
    }
    
    metrics->last_inference_time = cnn_time;
    metrics->total_inferences = 0; // Would be incremented in actual implementation
    metrics->average_inference_time = cnn_time; // Would be calculated from history
    metrics->peak_inference_time = cnn_time; // Would track maximum
    metrics->processor_utilization = 85; // Would be calculated from actual usage
    
    return EMG_CNN_OK;
}