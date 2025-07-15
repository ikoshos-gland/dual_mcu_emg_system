/**
 * @file ads1299_driver.c
 * @brief ADS1299 8-Channel EMG ADC Driver Implementation
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include "ads1299_driver.h"
#include <string.h>
#include <stdlib.h>

/* Private defines -----------------------------------------------------------*/
#define ADS1299_TIMEOUT_MS          1000
#define ADS1299_RESET_DELAY_MS      10
#define ADS1299_STARTUP_DELAY_MS    1000
#define ADS1299_COMMAND_DELAY_US    4       // tCLK = 4 * tCLK cycles
#define ADS1299_DRDY_TIMEOUT_MS     100

/* Private variables ---------------------------------------------------------*/
static uint8_t ads1299_tx_buffer[32];
static uint8_t ads1299_rx_buffer[32];
static bool ads1299_dma_complete = false;

/* Private function prototypes -----------------------------------------------*/
static EMG_StatusTypeDef ADS1299_SendCommand(ADS1299_HandleTypeDef *hads, uint8_t command);
static EMG_StatusTypeDef ADS1299_WaitForDRDY(ADS1299_HandleTypeDef *hads, uint32_t timeout_ms);
static EMG_StatusTypeDef ADS1299_ReadDataContinuous(ADS1299_HandleTypeDef *hads, ADS1299_DataTypeDef *data);
static void ADS1299_CS_Enable(ADS1299_HandleTypeDef *hads);
static void ADS1299_CS_Disable(ADS1299_HandleTypeDef *hads);
static void ADS1299_Reset_Assert(ADS1299_HandleTypeDef *hads);
static void ADS1299_Reset_Deassert(ADS1299_HandleTypeDef *hads);
static void ADS1299_Start_Assert(ADS1299_HandleTypeDef *hads);
static void ADS1299_Start_Deassert(ADS1299_HandleTypeDef *hads);
static bool ADS1299_IsDRDY(ADS1299_HandleTypeDef *hads);

/* Public functions ----------------------------------------------------------*/

/**
 * @brief Initialize ADS1299
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_Init(ADS1299_HandleTypeDef *hads)
{
    EMG_StatusTypeDef status;
    uint8_t device_id;
    
    if (hads == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Initialize buffer */
    hads->data_buffer = (ADS1299_DataTypeDef *)malloc(hads->buffer_size * sizeof(ADS1299_DataTypeDef));
    if (hads->data_buffer == NULL) {
        return EMG_ERROR_INIT;
    }
    
    hads->buffer_head = 0;
    hads->buffer_tail = 0;
    hads->is_running = false;
    
    /* Hardware reset */
    status = ADS1299_Reset(hads);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Wait for startup */
    ADS1299_DelayMs(ADS1299_STARTUP_DELAY_MS);
    
    /* Read device ID */
    status = ADS1299_ReadRegister(hads, ADS1299_REG_ID, &device_id);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Check device ID (should be 0x3E for ADS1299) */
    if ((device_id & 0xFE) != 0x3E) {
        return EMG_ERROR_INIT;
    }
    
    hads->device_id = device_id;
    
    /* Stop continuous mode */
    status = ADS1299_SendCommand(hads, ADS1299_CMD_SDATAC);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Configure device */
    status = ADS1299_WriteRegister(hads, ADS1299_REG_CONFIG1, ADS1299_CONFIG1_DR_16KSPS);
    if (status != EMG_OK) {
        return status;
    }
    
    status = ADS1299_WriteRegister(hads, ADS1299_REG_CONFIG2, 0x00);
    if (status != EMG_OK) {
        return status;
    }
    
    status = ADS1299_WriteRegister(hads, ADS1299_REG_CONFIG3, 
                                  ADS1299_CONFIG3_PD_REFBUF | ADS1299_CONFIG3_BIASREF_INT);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Initialize all channels to default configuration */
    for (uint8_t i = 0; i < EMG_CHANNELS; i++) {
        hads->channel_config[i] = ADS1299_CH_GAIN_12 | ADS1299_CH_INPUT_NORMAL;
        status = ADS1299_WriteRegister(hads, ADS1299_REG_CH1SET + i, hads->channel_config[i]);
        if (status != EMG_OK) {
            return status;
        }
    }
    
    hads->is_initialized = true;
    
    return EMG_OK;
}

/**
 * @brief De-initialize ADS1299
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_DeInit(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Stop if running */
    if (hads->is_running) {
        ADS1299_Stop(hads);
    }
    
    /* Reset device */
    ADS1299_Reset(hads);
    
    /* Free buffer */
    if (hads->data_buffer != NULL) {
        free(hads->data_buffer);
        hads->data_buffer = NULL;
    }
    
    hads->is_initialized = false;
    
    return EMG_OK;
}

/**
 * @brief Reset ADS1299
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_Reset(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Hardware reset */
    ADS1299_Reset_Assert(hads);
    ADS1299_DelayMs(ADS1299_RESET_DELAY_MS);
    ADS1299_Reset_Deassert(hads);
    
    /* Wait for reset to complete */
    ADS1299_DelayMs(ADS1299_RESET_DELAY_MS);
    
    /* Send software reset command */
    ADS1299_SendCommand(hads, ADS1299_CMD_RESET);
    
    /* Wait for reset to complete */
    ADS1299_DelayMs(ADS1299_RESET_DELAY_MS);
    
    return EMG_OK;
}

/**
 * @brief Start ADS1299 conversion
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_Start(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL || !hads->is_initialized) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Start conversion */
    ADS1299_Start_Assert(hads);
    
    /* Send start command */
    ADS1299_SendCommand(hads, ADS1299_CMD_START);
    
    hads->is_running = true;
    
    return EMG_OK;
}

/**
 * @brief Stop ADS1299 conversion
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_Stop(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    /* Stop conversion */
    ADS1299_Start_Deassert(hads);
    
    /* Send stop command */
    ADS1299_SendCommand(hads, ADS1299_CMD_STOP);
    
    hads->is_running = false;
    
    return EMG_OK;
}

/**
 * @brief Write register
 * @param hads: ADS1299 handle pointer
 * @param reg: Register address
 * @param value: Value to write
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_WriteRegister(ADS1299_HandleTypeDef *hads, uint8_t reg, uint8_t value)
{
    if (hads == NULL || hads->hspi == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    HAL_StatusTypeDef hal_status;
    
    ads1299_tx_buffer[0] = ADS1299_CMD_WREG | reg;
    ads1299_tx_buffer[1] = 0x00; // n = 0 (write 1 register)
    ads1299_tx_buffer[2] = value;
    
    ADS1299_CS_Enable(hads);
    
    hal_status = HAL_SPI_Transmit(hads->hspi, ads1299_tx_buffer, 3, ADS1299_TIMEOUT_MS);
    
    ADS1299_CS_Disable(hads);
    
    if (hal_status != HAL_OK) {
        return EMG_ERROR_SPI;
    }
    
    ADS1299_DelayUs(ADS1299_COMMAND_DELAY_US);
    
    return EMG_OK;
}

/**
 * @brief Read register
 * @param hads: ADS1299 handle pointer
 * @param reg: Register address
 * @param value: Pointer to store read value
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_ReadRegister(ADS1299_HandleTypeDef *hads, uint8_t reg, uint8_t *value)
{
    if (hads == NULL || hads->hspi == NULL || value == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    HAL_StatusTypeDef hal_status;
    
    ads1299_tx_buffer[0] = ADS1299_CMD_RREG | reg;
    ads1299_tx_buffer[1] = 0x00; // n = 0 (read 1 register)
    
    ADS1299_CS_Enable(hads);
    
    hal_status = HAL_SPI_Transmit(hads->hspi, ads1299_tx_buffer, 2, ADS1299_TIMEOUT_MS);
    if (hal_status != HAL_OK) {
        ADS1299_CS_Disable(hads);
        return EMG_ERROR_SPI;
    }
    
    hal_status = HAL_SPI_Receive(hads->hspi, ads1299_rx_buffer, 1, ADS1299_TIMEOUT_MS);
    
    ADS1299_CS_Disable(hads);
    
    if (hal_status != HAL_OK) {
        return EMG_ERROR_SPI;
    }
    
    *value = ads1299_rx_buffer[0];
    
    return EMG_OK;
}

/**
 * @brief Configure channel
 * @param hads: ADS1299 handle pointer
 * @param channel: Channel number (0-7)
 * @param config: Channel configuration
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_ConfigureChannel(ADS1299_HandleTypeDef *hads, uint8_t channel, uint8_t config)
{
    if (hads == NULL || channel >= EMG_CHANNELS) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    hads->channel_config[channel] = config;
    
    status = ADS1299_WriteRegister(hads, ADS1299_REG_CH1SET + channel, config);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief Set sampling rate
 * @param hads: ADS1299 handle pointer
 * @param sampling_rate: Sampling rate in Hz
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_SetSamplingRate(ADS1299_HandleTypeDef *hads, uint16_t sampling_rate)
{
    if (hads == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    uint8_t config_value;
    
    /* Select configuration based on sampling rate */
    switch (sampling_rate) {
        case 16000:
            config_value = ADS1299_CONFIG1_DR_16KSPS;
            break;
        case 8000:
            config_value = ADS1299_CONFIG1_DR_8KSPS;
            break;
        case 4000:
            config_value = ADS1299_CONFIG1_DR_4KSPS;
            break;
        case 2000:
            config_value = ADS1299_CONFIG1_DR_2KSPS;
            break;
        case 1000:
            config_value = ADS1299_CONFIG1_DR_1KSPS;
            break;
        case 500:
            config_value = ADS1299_CONFIG1_DR_500SPS;
            break;
        default:
            return EMG_ERROR_INVALID_PARAM;
    }
    
    status = ADS1299_WriteRegister(hads, ADS1299_REG_CONFIG1, config_value);
    if (status != EMG_OK) {
        return status;
    }
    
    hads->sampling_rate = sampling_rate;
    
    return EMG_OK;
}

/**
 * @brief Start continuous mode
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_StartContinuousMode(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL || !hads->is_initialized) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Start conversions */
    status = ADS1299_Start(hads);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Enable continuous mode */
    status = ADS1299_SendCommand(hads, ADS1299_CMD_RDATAC);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief Stop continuous mode
 * @param hads: ADS1299 handle pointer
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_StopContinuousMode(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    EMG_StatusTypeDef status;
    
    /* Stop continuous mode */
    status = ADS1299_SendCommand(hads, ADS1299_CMD_SDATAC);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Stop conversions */
    status = ADS1299_Stop(hads);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief Read data
 * @param hads: ADS1299 handle pointer
 * @param data: Pointer to data structure
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_ReadData(ADS1299_HandleTypeDef *hads, ADS1299_DataTypeDef *data)
{
    if (hads == NULL || data == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    if (!hads->is_running) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    return ADS1299_ReadDataContinuous(hads, data);
}

/**
 * @brief Check if data is ready
 * @param hads: ADS1299 handle pointer
 * @retval true if data is ready
 */
bool ADS1299_IsDataReady(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return false;
    }
    
    return !ADS1299_IsDRDY(hads);
}

/**
 * @brief Get buffer count
 * @param hads: ADS1299 handle pointer
 * @retval Number of samples in buffer
 */
uint16_t ADS1299_GetBufferCount(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return 0;
    }
    
    if (hads->buffer_head >= hads->buffer_tail) {
        return hads->buffer_head - hads->buffer_tail;
    } else {
        return (hads->buffer_size - hads->buffer_tail) + hads->buffer_head;
    }
}

/**
 * @brief Get data from buffer
 * @param hads: ADS1299 handle pointer
 * @param data: Pointer to data structure
 * @retval EMG_StatusTypeDef
 */
EMG_StatusTypeDef ADS1299_GetBufferData(ADS1299_HandleTypeDef *hads, ADS1299_DataTypeDef *data)
{
    if (hads == NULL || data == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    if (ADS1299_GetBufferCount(hads) == 0) {
        return EMG_ERROR_BUFFER_FULL;
    }
    
    *data = hads->data_buffer[hads->buffer_tail];
    hads->buffer_tail = (hads->buffer_tail + 1) % hads->buffer_size;
    
    return EMG_OK;
}

/**
 * @brief DRDY interrupt handler
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
void ADS1299_DRDY_IRQHandler(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL || !hads->is_running) {
        return;
    }
    
    ADS1299_DataTypeDef data;
    
    /* Read data */
    if (ADS1299_ReadDataContinuous(hads, &data) == EMG_OK) {
        /* Store in buffer */
        hads->data_buffer[hads->buffer_head] = data;
        hads->buffer_head = (hads->buffer_head + 1) % hads->buffer_size;
        
        /* Check for buffer overflow */
        if (hads->buffer_head == hads->buffer_tail) {
            hads->buffer_tail = (hads->buffer_tail + 1) % hads->buffer_size;
        }
        
        /* Call callback if available */
        if (hads->data_ready_callback != NULL) {
            hads->data_ready_callback(&data);
        }
    }
}

/**
 * @brief SPI transfer complete callback
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
void ADS1299_SPI_TxRxCpltCallback(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return;
    }
    
    ads1299_dma_complete = true;
}

/**
 * @brief SPI error callback
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
void ADS1299_SPI_ErrorCallback(ADS1299_HandleTypeDef *hads)
{
    if (hads == NULL) {
        return;
    }
    
    // Handle SPI error
    hads->is_running = false;
}

/* Private functions ---------------------------------------------------------*/

/**
 * @brief Send command to ADS1299
 * @param hads: ADS1299 handle pointer
 * @param command: Command to send
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef ADS1299_SendCommand(ADS1299_HandleTypeDef *hads, uint8_t command)
{
    if (hads == NULL || hads->hspi == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    HAL_StatusTypeDef hal_status;
    
    ADS1299_CS_Enable(hads);
    
    hal_status = HAL_SPI_Transmit(hads->hspi, &command, 1, ADS1299_TIMEOUT_MS);
    
    ADS1299_CS_Disable(hads);
    
    if (hal_status != HAL_OK) {
        return EMG_ERROR_SPI;
    }
    
    ADS1299_DelayUs(ADS1299_COMMAND_DELAY_US);
    
    return EMG_OK;
}

/**
 * @brief Read data in continuous mode
 * @param hads: ADS1299 handle pointer
 * @param data: Pointer to data structure
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef ADS1299_ReadDataContinuous(ADS1299_HandleTypeDef *hads, ADS1299_DataTypeDef *data)
{
    if (hads == NULL || hads->hspi == NULL || data == NULL) {
        return EMG_ERROR_INVALID_PARAM;
    }
    
    HAL_StatusTypeDef hal_status;
    uint8_t rx_data[27]; // 3 status bytes + 8 channels * 3 bytes
    
    /* Wait for DRDY */
    if (ADS1299_WaitForDRDY(hads, ADS1299_DRDY_TIMEOUT_MS) != EMG_OK) {
        return EMG_ERROR_TIMEOUT;
    }
    
    ADS1299_CS_Enable(hads);
    
    /* Read data using DMA if available */
    if (hads->hspi->hdmarx != NULL) {
        ads1299_dma_complete = false;
        hal_status = HAL_SPI_Receive_DMA(hads->hspi, rx_data, 27);
        
        if (hal_status != HAL_OK) {
            ADS1299_CS_Disable(hads);
            return EMG_ERROR_DMA;
        }
        
        /* Wait for DMA completion */
        uint32_t timeout = HAL_GetTick() + ADS1299_TIMEOUT_MS;
        while (!ads1299_dma_complete && HAL_GetTick() < timeout) {
            // Wait
        }
        
        if (!ads1299_dma_complete) {
            ADS1299_CS_Disable(hads);
            return EMG_ERROR_TIMEOUT;
        }
    } else {
        /* Use polling mode */
        hal_status = HAL_SPI_Receive(hads->hspi, rx_data, 27, ADS1299_TIMEOUT_MS);
        
        if (hal_status != HAL_OK) {
            ADS1299_CS_Disable(hads);
            return EMG_ERROR_SPI;
        }
    }
    
    ADS1299_CS_Disable(hads);
    
    /* Parse received data */
    data->status = rx_data[0];
    data->timestamp = HAL_GetTick() * 1000; // Convert to microseconds
    
    /* Convert 24-bit data to 32-bit signed */
    for (uint8_t i = 0; i < EMG_CHANNELS; i++) {
        uint32_t raw_data = (rx_data[3 + i * 3] << 16) | 
                           (rx_data[3 + i * 3 + 1] << 8) | 
                           rx_data[3 + i * 3 + 2];
        data->channel_data[i] = ADS1299_ConvertToSigned24(raw_data);
    }
    
    data->data_ready = true;
    
    return EMG_OK;
}

/**
 * @brief Wait for DRDY signal
 * @param hads: ADS1299 handle pointer
 * @param timeout_ms: Timeout in milliseconds
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef ADS1299_WaitForDRDY(ADS1299_HandleTypeDef *hads, uint32_t timeout_ms)
{
    uint32_t timeout = HAL_GetTick() + timeout_ms;
    
    while (!ADS1299_IsDataReady(hads)) {
        if (HAL_GetTick() >= timeout) {
            return EMG_ERROR_TIMEOUT;
        }
    }
    
    return EMG_OK;
}

/**
 * @brief Enable CS signal
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
static void ADS1299_CS_Enable(ADS1299_HandleTypeDef *hads)
{
    HAL_GPIO_WritePin(hads->cs_port, hads->cs_pin, GPIO_PIN_RESET);
}

/**
 * @brief Disable CS signal
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
static void ADS1299_CS_Disable(ADS1299_HandleTypeDef *hads)
{
    HAL_GPIO_WritePin(hads->cs_port, hads->cs_pin, GPIO_PIN_SET);
}

/**
 * @brief Assert reset signal
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
static void ADS1299_Reset_Assert(ADS1299_HandleTypeDef *hads)
{
    HAL_GPIO_WritePin(hads->reset_port, hads->reset_pin, GPIO_PIN_RESET);
}

/**
 * @brief Deassert reset signal
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
static void ADS1299_Reset_Deassert(ADS1299_HandleTypeDef *hads)
{
    HAL_GPIO_WritePin(hads->reset_port, hads->reset_pin, GPIO_PIN_SET);
}

/**
 * @brief Assert start signal
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
static void ADS1299_Start_Assert(ADS1299_HandleTypeDef *hads)
{
    HAL_GPIO_WritePin(hads->start_port, hads->start_pin, GPIO_PIN_SET);
}

/**
 * @brief Deassert start signal
 * @param hads: ADS1299 handle pointer
 * @retval None
 */
static void ADS1299_Start_Deassert(ADS1299_HandleTypeDef *hads)
{
    HAL_GPIO_WritePin(hads->start_port, hads->start_pin, GPIO_PIN_RESET);
}

/**
 * @brief Check DRDY signal
 * @param hads: ADS1299 handle pointer
 * @retval true if DRDY is low (data ready)
 */
static bool ADS1299_IsDRDY(ADS1299_HandleTypeDef *hads)
{
    return HAL_GPIO_ReadPin(hads->drdy_port, hads->drdy_pin) == GPIO_PIN_RESET;
}

/* Utility functions ---------------------------------------------------------*/

/**
 * @brief Delay in milliseconds
 * @param delay: Delay in milliseconds
 * @retval None
 */
void ADS1299_DelayMs(uint32_t delay)
{
    HAL_Delay(delay);
}

/**
 * @brief Delay in microseconds
 * @param delay: Delay in microseconds
 * @retval None
 */
void ADS1299_DelayUs(uint32_t delay)
{
    /* Simple delay loop - not precise but sufficient for short delays */
    for (volatile uint32_t i = 0; i < delay * 150; i++) {
        __NOP();
    }
}

/**
 * @brief Convert 24-bit unsigned to 32-bit signed
 * @param raw_data: 24-bit unsigned data
 * @retval 32-bit signed data
 */
int32_t ADS1299_ConvertToSigned24(uint32_t raw_data)
{
    /* Check sign bit (bit 23) */
    if (raw_data & 0x800000) {
        /* Negative number - sign extend */
        return (int32_t)(raw_data | 0xFF000000);
    } else {
        /* Positive number */
        return (int32_t)(raw_data & 0x7FFFFF);
    }
}

/**
 * @brief Convert raw data to voltage
 * @param raw_data: Raw 24-bit signed data
 * @param gain: Gain setting
 * @retval Voltage in microvolts
 */
float ADS1299_ConvertToVoltage(int32_t raw_data, uint8_t gain)
{
    float vref = 4.5f; // 4.5V reference
    float gain_value;
    
    /* Determine gain value */
    switch (gain & 0x70) {
        case ADS1299_CH_GAIN_1:
            gain_value = 1.0f;
            break;
        case ADS1299_CH_GAIN_2:
            gain_value = 2.0f;
            break;
        case ADS1299_CH_GAIN_4:
            gain_value = 4.0f;
            break;
        case ADS1299_CH_GAIN_6:
            gain_value = 6.0f;
            break;
        case ADS1299_CH_GAIN_8:
            gain_value = 8.0f;
            break;
        case ADS1299_CH_GAIN_12:
            gain_value = 12.0f;
            break;
        case ADS1299_CH_GAIN_24:
            gain_value = 24.0f;
            break;
        default:
            gain_value = 1.0f;
            break;
    }
    
    /* Convert to voltage in microvolts */
    return ((float)raw_data * vref * 1000000.0f) / (gain_value * 8388608.0f); // 2^23
}