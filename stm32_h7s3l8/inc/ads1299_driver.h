/**
 * @file ads1299_driver.h
 * @brief ADS1299 8-Channel EMG ADC Driver Header
 * @author Dual-MCU EMG System
 * @date 2025
 */

#ifndef ADS1299_DRIVER_H
#define ADS1299_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include <stdint.h>
#include <stdbool.h>

/* ADS1299 Register Definitions */
#define ADS1299_REG_ID              0x00
#define ADS1299_REG_CONFIG1         0x01
#define ADS1299_REG_CONFIG2         0x02
#define ADS1299_REG_CONFIG3         0x03
#define ADS1299_REG_LOFF            0x04
#define ADS1299_REG_CH1SET          0x05
#define ADS1299_REG_CH2SET          0x06
#define ADS1299_REG_CH3SET          0x07
#define ADS1299_REG_CH4SET          0x08
#define ADS1299_REG_CH5SET          0x09
#define ADS1299_REG_CH6SET          0x0A
#define ADS1299_REG_CH7SET          0x0B
#define ADS1299_REG_CH8SET          0x0C
#define ADS1299_REG_BIAS_SENSP      0x0D
#define ADS1299_REG_BIAS_SENSN      0x0E
#define ADS1299_REG_LOFF_SENSP      0x0F
#define ADS1299_REG_LOFF_SENSN      0x10
#define ADS1299_REG_LOFF_FLIP       0x11
#define ADS1299_REG_LOFF_STATP      0x12
#define ADS1299_REG_LOFF_STATN      0x13
#define ADS1299_REG_GPIO            0x14
#define ADS1299_REG_MISC1           0x15
#define ADS1299_REG_MISC2           0x16
#define ADS1299_REG_CONFIG4         0x17

/* ADS1299 SPI Commands */
#define ADS1299_CMD_WAKEUP          0x02
#define ADS1299_CMD_STANDBY         0x04
#define ADS1299_CMD_RESET           0x06
#define ADS1299_CMD_START           0x08
#define ADS1299_CMD_STOP            0x0A
#define ADS1299_CMD_RDATAC          0x10
#define ADS1299_CMD_SDATAC          0x11
#define ADS1299_CMD_RDATA           0x12
#define ADS1299_CMD_RREG            0x20
#define ADS1299_CMD_WREG            0x40

/* Configuration Values */
#define ADS1299_CONFIG1_DR_16KSPS   0x95    // 16kSPS, CLK output enabled
#define ADS1299_CONFIG1_DR_8KSPS    0x94    // 8kSPS
#define ADS1299_CONFIG1_DR_4KSPS    0x93    // 4kSPS
#define ADS1299_CONFIG1_DR_2KSPS    0x92    // 2kSPS
#define ADS1299_CONFIG1_DR_1KSPS    0x91    // 1kSPS
#define ADS1299_CONFIG1_DR_500SPS   0x90    // 500SPS

#define ADS1299_CONFIG2_INT_TEST    0x10    // Internal test signal
#define ADS1299_CONFIG2_TEST_AMP    0x04    // Test signal amplitude
#define ADS1299_CONFIG2_TEST_FREQ   0x03    // Test signal frequency

#define ADS1299_CONFIG3_PD_REFBUF   0x80    // Power down reference buffer
#define ADS1299_CONFIG3_BIAS_MEAS   0x10    // BIAS measurement
#define ADS1299_CONFIG3_BIASREF_INT 0x08    // Internal BIAS reference
#define ADS1299_CONFIG3_PD_BIAS     0x04    // Power down BIAS buffer
#define ADS1299_CONFIG3_BIAS_LOFF   0x02    // BIAS lead-off sensing
#define ADS1299_CONFIG3_BIAS_STAT   0x01    // BIAS status

/* Channel Settings */
#define ADS1299_CH_POWERDOWN        0x80    // Power down channel
#define ADS1299_CH_GAIN_1           0x00    // Gain = 1
#define ADS1299_CH_GAIN_2           0x10    // Gain = 2
#define ADS1299_CH_GAIN_4           0x20    // Gain = 4
#define ADS1299_CH_GAIN_6           0x30    // Gain = 6
#define ADS1299_CH_GAIN_8           0x40    // Gain = 8
#define ADS1299_CH_GAIN_12          0x50    // Gain = 12
#define ADS1299_CH_GAIN_24          0x60    // Gain = 24

#define ADS1299_CH_SRB2             0x08    // SRB2 connection
#define ADS1299_CH_INPUT_NORMAL     0x00    // Normal electrode input
#define ADS1299_CH_INPUT_SHORTED    0x01    // Shorted
#define ADS1299_CH_INPUT_BIAS_MEAS  0x02    // BIAS measurement
#define ADS1299_CH_INPUT_MVDD       0x03    // MVDD supply
#define ADS1299_CH_INPUT_TEMP       0x04    // Temperature sensor
#define ADS1299_CH_INPUT_TEST       0x05    // Test signal
#define ADS1299_CH_INPUT_BIAS_DRP   0x06    // BIAS drive positive
#define ADS1299_CH_INPUT_BIAS_DRN   0x07    // BIAS drive negative

/* Data Structure */
typedef struct {
    int32_t channel_data[EMG_CHANNELS];     // 24-bit signed data
    uint32_t timestamp;                      // Timestamp in microseconds
    uint8_t status;                         // Status byte
    bool data_ready;                        // Data ready flag
} ADS1299_DataTypeDef;

typedef struct {
    SPI_HandleTypeDef *hspi;                // SPI handle
    GPIO_TypeDef *cs_port;                  // CS port
    uint16_t cs_pin;                        // CS pin
    GPIO_TypeDef *drdy_port;                // DRDY port
    uint16_t drdy_pin;                      // DRDY pin
    GPIO_TypeDef *reset_port;               // Reset port
    uint16_t reset_pin;                     // Reset pin
    GPIO_TypeDef *start_port;               // Start port
    uint16_t start_pin;                     // Start pin
    
    uint8_t device_id;                      // Device ID
    uint16_t sampling_rate;                 // Sampling rate
    uint8_t channel_config[EMG_CHANNELS];   // Channel configurations
    bool is_initialized;                    // Initialization flag
    bool is_running;                        // Running flag
    
    ADS1299_DataTypeDef *data_buffer;       // Data buffer
    uint16_t buffer_size;                   // Buffer size
    uint16_t buffer_head;                   // Buffer head pointer
    uint16_t buffer_tail;                   // Buffer tail pointer
    
    void (*data_ready_callback)(ADS1299_DataTypeDef *data);  // Callback function
} ADS1299_HandleTypeDef;

/* Function Prototypes */
EMG_StatusTypeDef ADS1299_Init(ADS1299_HandleTypeDef *hads);
EMG_StatusTypeDef ADS1299_DeInit(ADS1299_HandleTypeDef *hads);
EMG_StatusTypeDef ADS1299_Reset(ADS1299_HandleTypeDef *hads);
EMG_StatusTypeDef ADS1299_Start(ADS1299_HandleTypeDef *hads);
EMG_StatusTypeDef ADS1299_Stop(ADS1299_HandleTypeDef *hads);

EMG_StatusTypeDef ADS1299_WriteRegister(ADS1299_HandleTypeDef *hads, uint8_t reg, uint8_t value);
EMG_StatusTypeDef ADS1299_ReadRegister(ADS1299_HandleTypeDef *hads, uint8_t reg, uint8_t *value);
EMG_StatusTypeDef ADS1299_ReadMultipleRegisters(ADS1299_HandleTypeDef *hads, uint8_t start_reg, uint8_t *buffer, uint8_t count);

EMG_StatusTypeDef ADS1299_ConfigureChannel(ADS1299_HandleTypeDef *hads, uint8_t channel, uint8_t config);
EMG_StatusTypeDef ADS1299_SetSamplingRate(ADS1299_HandleTypeDef *hads, uint16_t sampling_rate);
EMG_StatusTypeDef ADS1299_SetGain(ADS1299_HandleTypeDef *hads, uint8_t channel, uint8_t gain);

EMG_StatusTypeDef ADS1299_ReadData(ADS1299_HandleTypeDef *hads, ADS1299_DataTypeDef *data);
EMG_StatusTypeDef ADS1299_StartContinuousMode(ADS1299_HandleTypeDef *hads);
EMG_StatusTypeDef ADS1299_StopContinuousMode(ADS1299_HandleTypeDef *hads);

bool ADS1299_IsDataReady(ADS1299_HandleTypeDef *hads);
uint16_t ADS1299_GetBufferCount(ADS1299_HandleTypeDef *hads);
EMG_StatusTypeDef ADS1299_GetBufferData(ADS1299_HandleTypeDef *hads, ADS1299_DataTypeDef *data);

void ADS1299_DRDY_IRQHandler(ADS1299_HandleTypeDef *hads);
void ADS1299_SPI_TxRxCpltCallback(ADS1299_HandleTypeDef *hads);
void ADS1299_SPI_ErrorCallback(ADS1299_HandleTypeDef *hads);

/* Utility Functions */
void ADS1299_DelayMs(uint32_t delay);
void ADS1299_DelayUs(uint32_t delay);
int32_t ADS1299_ConvertToSigned24(uint32_t raw_data);
float ADS1299_ConvertToVoltage(int32_t raw_data, uint8_t gain);

#ifdef __cplusplus
}
#endif

#endif /* ADS1299_DRIVER_H */