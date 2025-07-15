/**
 * @file main.h
 * @brief Main header file for STM32 H7S3L8 EMG Processing System
 * @author Dual-MCU EMG System
 * @date 2025
 */

#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7rsxx_hal.h"
#include "stm32h7rsxx_ll_rcc.h"
#include "stm32h7rsxx_ll_utils.h"
#include "stm32h7rsxx_ll_system.h"
#include "stm32h7rsxx_ll_gpio.h"
#include "stm32h7rsxx_ll_exti.h"
#include "stm32h7rsxx_ll_bus.h"
#include "stm32h7rsxx_ll_cortex.h"
#include "stm32h7rsxx_ll_pwr.h"
#include "stm32h7rsxx_ll_dma.h"
#include "stm32h7rsxx_ll_spi.h"
#include "stm32h7rsxx_ll_tim.h"
#include "stm32h7rsxx_ll_usart.h"

/* System Configuration */
#define SYSTEM_CLOCK_FREQUENCY      600000000U  // 600MHz
#define HCLK_FREQUENCY              300000000U  // 300MHz
#define PCLK1_FREQUENCY             150000000U  // 150MHz
#define PCLK2_FREQUENCY             150000000U  // 150MHz

/* ADS1299 Configuration */
#define ADS1299_SPI_INSTANCE        SPI1
#define ADS1299_SPI_BAUDRATE        20000000U   // 20MHz
#define ADS1299_CS_PORT             GPIOA
#define ADS1299_CS_PIN              GPIO_PIN_4
#define ADS1299_DRDY_PORT           GPIOA
#define ADS1299_DRDY_PIN            GPIO_PIN_5
#define ADS1299_RESET_PORT          GPIOA
#define ADS1299_RESET_PIN           GPIO_PIN_6
#define ADS1299_START_PORT          GPIOA
#define ADS1299_START_PIN           GPIO_PIN_7

/* MAX78000 Communication Configuration */
#define MAX78000_SPI_INSTANCE       SPI2
#define MAX78000_SPI_BAUDRATE       10000000U   // 10MHz
#define MAX78000_CS_PORT            GPIOB
#define MAX78000_CS_PIN             GPIO_PIN_12
#define MAX78000_IRQ_PORT           GPIOB
#define MAX78000_IRQ_PIN            GPIO_PIN_13

/* EMG Processing Configuration */
#define EMG_CHANNELS                8
#define EMG_SAMPLING_RATE           16000       // 16kSPS
#define EMG_BUFFER_SIZE             1024        // Circular buffer size
#define EMG_WINDOW_SIZE             512         // Processing window size (32ms @ 16kSPS)
#define EMG_OVERLAP                 256         // 50% overlap
#define EMG_FEATURE_UPDATE_RATE     100         // 100Hz feature updates

/* DSP Configuration */
#define DSP_BANDPASS_LOW_FREQ       20          // 20Hz highpass
#define DSP_BANDPASS_HIGH_FREQ      450         // 450Hz lowpass
#define DSP_NOTCH_FREQ              50          // 50Hz notch filter
#define DSP_RMS_WINDOW_SIZE         50          // 50ms RMS window
#define DSP_FEATURE_VECTOR_SIZE     64          // Features per window

/* DMA Configuration */
#define DMA_INSTANCE                HPDMA1
#define DMA_CHANNEL_ADS1299         0
#define DMA_CHANNEL_MAX78000_TX     1
#define DMA_CHANNEL_MAX78000_RX     2

/* Memory Configuration */
#define SRAM_AXI_BASE               0x24000000U
#define SRAM_D1_BASE                0x30000000U
#define SRAM_D2_BASE                0x30040000U
#define SRAM_D3_BASE                0x38000000U

/* Error Codes */
typedef enum {
    EMG_OK = 0,
    EMG_ERROR_INIT,
    EMG_ERROR_SPI,
    EMG_ERROR_DMA,
    EMG_ERROR_TIMEOUT,
    EMG_ERROR_INVALID_PARAM,
    EMG_ERROR_BUFFER_FULL,
    EMG_ERROR_PROCESSING
} EMG_StatusTypeDef;

/* System States */
typedef enum {
    SYSTEM_INIT = 0,
    SYSTEM_READY,
    SYSTEM_RUNNING,
    SYSTEM_ERROR,
    SYSTEM_SUSPEND
} SystemStateTypeDef;

/* Function Prototypes */
void SystemClock_Config(void);
void MPU_Config(void);
void Error_Handler(void);
void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported variables */
extern SystemStateTypeDef system_state;

#ifdef __cplusplus
}
#endif

#endif /* MAIN_H */