/**
 * @file emg_communication.h
 * @brief MAX78000 Communication Interface Header
 * @author Dual-MCU EMG System
 * @date 2025
 */

#ifndef EMG_COMMUNICATION_H
#define EMG_COMMUNICATION_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include <stdbool.h>
#include "mxc_device.h"
#include "spi.h"
#include "gpio.h"
#include "emg_cnn.h"

/* Communication Status */
typedef enum {
    EMG_COMM_OK = 0,                            // Success
    EMG_COMM_ERROR_INVALID_PARAM,               // Invalid parameter
    EMG_COMM_ERROR_NOT_INITIALIZED,             // Not initialized
    EMG_COMM_ERROR_SPI_INIT,                    // SPI initialization failed
    EMG_COMM_ERROR_SPI_TRANSMIT,                // SPI transmit failed
    EMG_COMM_ERROR_SPI_RECEIVE,                 // SPI receive failed
    EMG_COMM_ERROR_SPI_CALLBACK,                // SPI callback error
    EMG_COMM_ERROR_TIMEOUT,                     // Timeout error
    EMG_COMM_ERROR_CHECKSUM,                    // Checksum error
    EMG_COMM_ERROR_INVALID_SYNC,                // Invalid sync byte
    EMG_COMM_ERROR_UNKNOWN_PACKET,              // Unknown packet type
    EMG_COMM_ERROR_BUFFER_FULL                  // Buffer full
} EMG_Communication_Status;

/* Communication Statistics */
typedef struct {
    uint32_t packets_received;                  // Total packets received
    uint32_t packets_sent;                      // Total packets sent
    uint32_t communication_errors;              // Communication errors
    uint32_t uptime;                            // System uptime in seconds
} EMG_Communication_Stats;

/* Callback Function Types */
typedef void (*EMG_Communication_FeatureCallback)(const int8_t *features, uint16_t size);
typedef void (*EMG_Communication_ErrorCallback)(uint8_t error_code);

/* Communication Configuration */
typedef struct {
    mxc_spi_regs_t *spi_instance;              // SPI instance
    mxc_gpio_regs_t *cs_port;                  // CS GPIO port
    uint32_t cs_pin;                           // CS GPIO pin
    mxc_gpio_regs_t *irq_port;                 // IRQ GPIO port
    uint32_t irq_pin;                          // IRQ GPIO pin
    uint32_t baudrate;                         // SPI baudrate
    EMG_Communication_FeatureCallback feature_callback;    // Feature data callback
    EMG_Communication_ErrorCallback error_callback;        // Error callback
} EMG_Communication_Config;

/* Function Prototypes */

/* Initialization and Configuration */
EMG_Communication_Status EMG_Communication_Init(const EMG_Communication_Config *config);
EMG_Communication_Status EMG_Communication_DeInit(void);
EMG_Communication_Status EMG_Communication_Reset(void);

/* Communication Functions */
void EMG_Communication_Process(void);
EMG_Communication_Status EMG_Communication_SendClassification(const EMG_CNN_ResultTypeDef *result);
EMG_Communication_Status EMG_Communication_SendHeartbeat(void);
EMG_Communication_Status EMG_Communication_SendStatus(void);

/* Statistics and Monitoring */
EMG_Communication_Status EMG_Communication_GetStats(EMG_Communication_Stats *stats);

#ifdef __cplusplus
}
#endif

#endif /* EMG_COMMUNICATION_H */