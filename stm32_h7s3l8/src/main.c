/**
 * @file main.c
 * @brief Main application file for STM32 H7S3L8 EMG Processing System
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "ads1299_driver.h"
#include "emg_processing.h"
#include "mcu_communication.h"

/* Private includes ----------------------------------------------------------*/
#include <string.h>
#include <stdio.h>

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
/* Private macro -------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/

/* HAL Handles */
SPI_HandleTypeDef hspi1;              // ADS1299 SPI
SPI_HandleTypeDef hspi2;              // MAX78000 SPI
TIM_HandleTypeDef htim2;              // System timer
DMA_HandleTypeDef hdma_spi1_rx;       // ADS1299 DMA
DMA_HandleTypeDef hdma_spi2_tx;       // MAX78000 TX DMA
DMA_HandleTypeDef hdma_spi2_rx;       // MAX78000 RX DMA

/* Application Handles */
ADS1299_HandleTypeDef hads1299;       // ADS1299 handle
EMG_ProcessingTypeDef hemg;           // EMG processing handle
MCU_CommHandleTypeDef hcomm;          // Communication handle

/* System State */
SystemStateTypeDef system_state = SYSTEM_INIT;

/* Private function prototypes -----------------------------------------------*/
static void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_SPI1_Init(void);
static void MX_SPI2_Init(void);
static void MX_TIM2_Init(void);

static EMG_StatusTypeDef System_Init(void);
static EMG_StatusTypeDef ADS1299_Setup(void);
static EMG_StatusTypeDef EMG_Processing_Setup(void);
static EMG_StatusTypeDef Communication_Setup(void);

static void System_Error_Handler(EMG_StatusTypeDef error);
static void System_Status_Update(void);

/* Application callback functions */
static void ADS1299_DataReadyCallback(ADS1299_DataTypeDef *data);
static void EMG_FeatureReadyCallback(EMG_FeatureVectorTypeDef *features);
static void MCU_Comm_PacketReceivedCallback(MCU_PacketTypeDef *packet);
static void MCU_Comm_ErrorCallback(MCU_StatusTypeDef error);

/* Private user code ---------------------------------------------------------*/

/**
 * @brief The application entry point.
 * @retval int
 */
int main(void)
{
    EMG_StatusTypeDef status;
    
    /* Configure the system */
    MPU_Config();
    
    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();
    
    /* Configure the system clock */
    SystemClock_Config();
    
    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_SPI1_Init();
    MX_SPI2_Init();
    MX_TIM2_Init();
    
    /* Initialize the system */
    status = System_Init();
    if (status != EMG_OK) {
        System_Error_Handler(status);
    }
    
    /* Set system state to ready */
    system_state = SYSTEM_READY;
    
    /* Start the system timer */
    HAL_TIM_Base_Start_IT(&htim2);
    
    /* Main application loop */
    while (1)
    {
        /* Process any pending communication */
        MCU_Comm_Process(&hcomm);
        
        /* Update system status */
        System_Status_Update();
        
        /* Check for system errors */
        if (system_state == SYSTEM_ERROR) {
            /* Handle system error */
            HAL_Delay(1000);
            system_state = SYSTEM_READY;
        }
        
        /* Small delay to prevent busy waiting */
        HAL_Delay(1);
    }
}

/**
 * @brief System initialization
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef System_Init(void)
{
    EMG_StatusTypeDef status;
    
    /* Initialize ADS1299 */
    status = ADS1299_Setup();
    if (status != EMG_OK) {
        return status;
    }
    
    /* Initialize EMG processing */
    status = EMG_Processing_Setup();
    if (status != EMG_OK) {
        return status;
    }
    
    /* Initialize communication */
    status = Communication_Setup();
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief ADS1299 setup and configuration
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef ADS1299_Setup(void)
{
    EMG_StatusTypeDef status;
    
    /* Configure ADS1299 handle */
    hads1299.hspi = &hspi1;
    hads1299.cs_port = ADS1299_CS_PORT;
    hads1299.cs_pin = ADS1299_CS_PIN;
    hads1299.drdy_port = ADS1299_DRDY_PORT;
    hads1299.drdy_pin = ADS1299_DRDY_PIN;
    hads1299.reset_port = ADS1299_RESET_PORT;
    hads1299.reset_pin = ADS1299_RESET_PIN;
    hads1299.start_port = ADS1299_START_PORT;
    hads1299.start_pin = ADS1299_START_PIN;
    hads1299.sampling_rate = EMG_SAMPLING_RATE;
    hads1299.data_ready_callback = ADS1299_DataReadyCallback;
    
    /* Initialize ADS1299 */
    status = ADS1299_Init(&hads1299);
    if (status != EMG_OK) {
        return status;
    }
    
    /* Configure all channels for EMG acquisition */
    for (uint8_t channel = 0; channel < EMG_CHANNELS; channel++) {
        status = ADS1299_ConfigureChannel(&hads1299, channel, 
                                         ADS1299_CH_GAIN_12 | ADS1299_CH_INPUT_NORMAL);
        if (status != EMG_OK) {
            return status;
        }
    }
    
    /* Set sampling rate */
    status = ADS1299_SetSamplingRate(&hads1299, EMG_SAMPLING_RATE);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief EMG processing setup
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef EMG_Processing_Setup(void)
{
    EMG_StatusTypeDef status;
    
    /* Configure EMG processing handle */
    hemg.feature_ready_callback = EMG_FeatureReadyCallback;
    hemg.error_callback = System_Error_Handler;
    
    /* Initialize EMG processing */
    status = EMG_Processing_Init(&hemg);
    if (status != EMG_OK) {
        return status;
    }
    
    return EMG_OK;
}

/**
 * @brief Communication setup
 * @retval EMG_StatusTypeDef
 */
static EMG_StatusTypeDef Communication_Setup(void)
{
    MCU_StatusTypeDef status;
    
    /* Configure communication handle */
    hcomm.hw_handle = &hspi2;
    hcomm.packet_received_callback = MCU_Comm_PacketReceivedCallback;
    hcomm.error_callback = MCU_Comm_ErrorCallback;
    
    /* Initialize communication */
    status = MCU_Comm_Init(&hcomm, &hspi2);
    if (status != MCU_STATUS_OK) {
        return EMG_ERROR_INIT;
    }
    
    /* Start communication */
    status = MCU_Comm_Start(&hcomm);
    if (status != MCU_STATUS_OK) {
        return EMG_ERROR_INIT;
    }
    
    return EMG_OK;
}

/**
 * @brief ADS1299 data ready callback
 * @param data: Pointer to ADS1299 data structure
 */
static void ADS1299_DataReadyCallback(ADS1299_DataTypeDef *data)
{
    if (system_state == SYSTEM_RUNNING) {
        /* Process the EMG sample */
        EMG_ProcessSample(&hemg, data);
    }
}

/**
 * @brief EMG feature ready callback
 * @param features: Pointer to feature vector
 */
static void EMG_FeatureReadyCallback(EMG_FeatureVectorTypeDef *features)
{
    if (system_state == SYSTEM_RUNNING) {
        /* Create feature data packet */
        MCU_FeatureDataTypeDef feature_packet;
        feature_packet.timestamp = features->timestamp;
        feature_packet.channel_count = EMG_CHANNELS;
        feature_packet.feature_count = FEATURE_TOTAL_PER_CHANNEL;
        feature_packet.data_format = 1; // int8 format
        
        /* Copy normalized features */
        memcpy(feature_packet.features, features->normalized_features, 
               sizeof(feature_packet.features));
        
        /* Send features to MAX78000 */
        MCU_Comm_SendFeatureData(&hcomm, &feature_packet);
    }
}

/**
 * @brief Communication packet received callback
 * @param packet: Pointer to received packet
 */
static void MCU_Comm_PacketReceivedCallback(MCU_PacketTypeDef *packet)
{
    switch (packet->header.packet_type) {
        case MCU_PACKET_CLASSIFICATION:
            /* Handle classification result */
            // Process classification result from MAX78000
            break;
            
        case MCU_PACKET_CONTROL:
            /* Handle control command */
            MCU_ControlTypeDef *control = &packet->payload.control;
            switch (control->command) {
                case MCU_CMD_START_INFERENCE:
                    ADS1299_StartContinuousMode(&hads1299);
                    EMG_Processing_Start(&hemg);
                    system_state = SYSTEM_RUNNING;
                    break;
                    
                case MCU_CMD_STOP_INFERENCE:
                    ADS1299_StopContinuousMode(&hads1299);
                    EMG_Processing_Stop(&hemg);
                    system_state = SYSTEM_READY;
                    break;
                    
                case MCU_CMD_RESET_SYSTEM:
                    NVIC_SystemReset();
                    break;
                    
                default:
                    break;
            }
            break;
            
        default:
            break;
    }
}

/**
 * @brief Communication error callback
 * @param error: Error code
 */
static void MCU_Comm_ErrorCallback(MCU_StatusTypeDef error)
{
    /* Handle communication error */
    system_state = SYSTEM_ERROR;
}

/**
 * @brief System error handler
 * @param error: Error code
 */
static void System_Error_Handler(EMG_StatusTypeDef error)
{
    system_state = SYSTEM_ERROR;
    
    /* Log error and take appropriate action */
    switch (error) {
        case EMG_ERROR_SPI:
            /* Reinitialize SPI */
            MX_SPI1_Init();
            MX_SPI2_Init();
            break;
            
        case EMG_ERROR_DMA:
            /* Reinitialize DMA */
            MX_DMA_Init();
            break;
            
        case EMG_ERROR_TIMEOUT:
            /* Reset communication */
            MCU_Comm_Stop(&hcomm);
            MCU_Comm_Start(&hcomm);
            break;
            
        default:
            break;
    }
}

/**
 * @brief System status update
 */
static void System_Status_Update(void)
{
    static uint32_t last_heartbeat = 0;
    uint32_t current_time = HAL_GetTick();
    
    /* Send heartbeat every second */
    if (current_time - last_heartbeat >= MCU_COMM_HEARTBEAT_INTERVAL) {
        MCU_Comm_SendHeartbeat(&hcomm);
        last_heartbeat = current_time;
    }
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
static void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    
    /** Configure the main internal regulator output voltage
     */
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    
    /** Initializes the RCC Oscillators according to the specified parameters
     * in the RCC_OscInitTypeDef structure.
     */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 2;
    RCC_OscInitStruct.PLL.PLLN = 150;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = 4;
    RCC_OscInitStruct.PLL.PLLR = 2;
    
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }
    
    /** Initializes the CPU, AHB and APB buses clocks
     */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;
    
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
        Error_Handler();
    }
}

/**
 * @brief MPU Configuration
 * @retval None
 */
static void MPU_Config(void)
{
    MPU_Region_InitTypeDef MPU_InitStruct = {0};
    
    /* Disables the MPU */
    HAL_MPU_Disable();
    
    /** Initializes and configures the Region and the memory to be protected
     */
    MPU_InitStruct.Enable = MPU_REGION_ENABLE;
    MPU_InitStruct.Number = MPU_REGION_NUMBER0;
    MPU_InitStruct.BaseAddress = 0x0;
    MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
    MPU_InitStruct.SubRegionDisable = 0x87;
    MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
    MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
    MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
    MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
    MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
    MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;
    
    HAL_MPU_ConfigRegion(&MPU_InitStruct);
    
    /* Enables the MPU */
    HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();
    
    /* Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(ADS1299_CS_PORT, ADS1299_CS_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADS1299_RESET_PORT, ADS1299_RESET_PIN, GPIO_PIN_SET);
    HAL_GPIO_WritePin(ADS1299_START_PORT, ADS1299_START_PIN, GPIO_PIN_LOW);
    HAL_GPIO_WritePin(MAX78000_CS_PORT, MAX78000_CS_PIN, GPIO_PIN_SET);
    
    /* Configure ADS1299 CS pin */
    GPIO_InitStruct.Pin = ADS1299_CS_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(ADS1299_CS_PORT, &GPIO_InitStruct);
    
    /* Configure ADS1299 RESET pin */
    GPIO_InitStruct.Pin = ADS1299_RESET_PIN;
    HAL_GPIO_Init(ADS1299_RESET_PORT, &GPIO_InitStruct);
    
    /* Configure ADS1299 START pin */
    GPIO_InitStruct.Pin = ADS1299_START_PIN;
    HAL_GPIO_Init(ADS1299_START_PORT, &GPIO_InitStruct);
    
    /* Configure ADS1299 DRDY pin */
    GPIO_InitStruct.Pin = ADS1299_DRDY_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(ADS1299_DRDY_PORT, &GPIO_InitStruct);
    
    /* Configure MAX78000 CS pin */
    GPIO_InitStruct.Pin = MAX78000_CS_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(MAX78000_CS_PORT, &GPIO_InitStruct);
    
    /* Configure MAX78000 IRQ pin */
    GPIO_InitStruct.Pin = MAX78000_IRQ_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
    GPIO_InitStruct.Pull = GPIO_PULLDOWN;
    HAL_GPIO_Init(MAX78000_IRQ_PORT, &GPIO_InitStruct);
    
    /* EXTI interrupt init */
    HAL_NVIC_SetPriority(EXTI5_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(EXTI5_IRQn);
    
    HAL_NVIC_SetPriority(EXTI13_IRQn, 1, 0);
    HAL_NVIC_EnableIRQ(EXTI13_IRQn);
}

/* Rest of the HAL initialization functions would go here */
/* MX_DMA_Init(), MX_SPI1_Init(), MX_SPI2_Init(), MX_TIM2_Init() */

/**
 * @brief This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void)
{
    /* User can add his own implementation to report the HAL error return state */
    __disable_irq();
    while (1) {
        /* Error indication */
    }
}

/**
 * @brief EXTI line detection callbacks.
 * @param GPIO_Pin: Specifies the pins connected EXTI line
 * @retval None
 */
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == ADS1299_DRDY_PIN) {
        /* ADS1299 data ready interrupt */
        ADS1299_DRDY_IRQHandler(&hads1299);
    } else if (GPIO_Pin == MAX78000_IRQ_PIN) {
        /* MAX78000 interrupt */
        // Handle MAX78000 interrupt
    }
}

/**
 * @brief TIM2 interrupt callback
 * @param htim: TIM handle
 * @retval None
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM2) {
        /* System tick callback */
        // Handle system timing
    }
}

#ifdef USE_FULL_ASSERT
/**
 * @brief Reports the name of the source file and the source line number
 *        where the assert_param error has occurred.
 * @param file: pointer to the source file name
 * @param line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line)
{
    /* User can add his own implementation to report the file name and line number,
       ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
}
#endif /* USE_FULL_ASSERT */