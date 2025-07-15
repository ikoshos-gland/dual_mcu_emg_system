/**
 * @file stm32_hal_config.c
 * @brief STM32 HAL Configuration for EMG System
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* External variables --------------------------------------------------------*/
extern DMA_HandleTypeDef hdma_spi1_rx;
extern DMA_HandleTypeDef hdma_spi2_tx;
extern DMA_HandleTypeDef hdma_spi2_rx;

/**
 * @brief SPI1 Initialization Function (ADS1299)
 * @param None
 * @retval None
 */
void MX_SPI1_Init(void)
{
    extern SPI_HandleTypeDef hspi1;
    
    /* SPI1 parameter configuration*/
    hspi1.Instance = SPI1;
    hspi1.Init.Mode = SPI_MODE_MASTER;
    hspi1.Init.Direction = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi1.Init.CLKPhase = SPI_PHASE_2EDGE;
    hspi1.Init.NSS = SPI_NSS_SOFT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_16;  // ~18.75 MHz for 300MHz PCLK
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    hspi1.Init.CRCPolynomial = 0x0;
    hspi1.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
    hspi1.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
    hspi1.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
    hspi1.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
    hspi1.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
    hspi1.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
    hspi1.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
    hspi1.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
    hspi1.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
    hspi1.Init.IOSwap = SPI_IO_SWAP_DISABLE;
    
    if (HAL_SPI_Init(&hspi1) != HAL_OK) {
        Error_Handler();
    }
}

/**
 * @brief SPI2 Initialization Function (MAX78000)
 * @param None
 * @retval None
 */
void MX_SPI2_Init(void)
{
    extern SPI_HandleTypeDef hspi2;
    
    /* SPI2 parameter configuration*/
    hspi2.Instance = SPI2;
    hspi2.Init.Mode = SPI_MODE_MASTER;
    hspi2.Init.Direction = SPI_DIRECTION_2LINES;
    hspi2.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi2.Init.CLKPolarity = SPI_POLARITY_LOW;
    hspi2.Init.CLKPhase = SPI_PHASE_1EDGE;
    hspi2.Init.NSS = SPI_NSS_SOFT;
    hspi2.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_32;  // ~9.375 MHz for 300MHz PCLK
    hspi2.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi2.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi2.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
    hspi2.Init.CRCPolynomial = 0x0;
    hspi2.Init.NSSPMode = SPI_NSS_PULSE_DISABLE;
    hspi2.Init.NSSPolarity = SPI_NSS_POLARITY_LOW;
    hspi2.Init.FifoThreshold = SPI_FIFO_THRESHOLD_01DATA;
    hspi2.Init.TxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
    hspi2.Init.RxCRCInitializationPattern = SPI_CRC_INITIALIZATION_ALL_ZERO_PATTERN;
    hspi2.Init.MasterSSIdleness = SPI_MASTER_SS_IDLENESS_00CYCLE;
    hspi2.Init.MasterInterDataIdleness = SPI_MASTER_INTERDATA_IDLENESS_00CYCLE;
    hspi2.Init.MasterReceiverAutoSusp = SPI_MASTER_RX_AUTOSUSP_DISABLE;
    hspi2.Init.MasterKeepIOState = SPI_MASTER_KEEP_IO_STATE_DISABLE;
    hspi2.Init.IOSwap = SPI_IO_SWAP_DISABLE;
    
    if (HAL_SPI_Init(&hspi2) != HAL_OK) {
        Error_Handler();
    }
}

/**
 * @brief TIM2 Initialization Function (System Timer)
 * @param None
 * @retval None
 */
void MX_TIM2_Init(void)
{
    extern TIM_HandleTypeDef htim2;
    
    TIM_ClockConfigTypeDef sClockSourceConfig = {0};
    TIM_MasterConfigTypeDef sMasterConfig = {0};
    
    htim2.Instance = TIM2;
    htim2.Init.Prescaler = 59999;           // 600MHz / 60000 = 10kHz
    htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim2.Init.Period = 9;                  // 10kHz / 10 = 1kHz (1ms)
    htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    
    if (HAL_TIM_Base_Init(&htim2) != HAL_OK) {
        Error_Handler();
    }
    
    sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
    if (HAL_TIM_ConfigClockSource(&htim2, &sClockSourceConfig) != HAL_OK) {
        Error_Handler();
    }
    
    sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
    sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
    if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK) {
        Error_Handler();
    }
}

/**
 * @brief DMA initialization function
 * @param None
 * @retval None
 */
void MX_DMA_Init(void)
{
    /* DMA controller clock enable */
    __HAL_RCC_HPDMA1_CLK_ENABLE();
    
    /* DMA interrupt init */
    /* HPDMA1_Channel0_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(HPDMA1_Channel0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(HPDMA1_Channel0_IRQn);
    
    /* HPDMA1_Channel1_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(HPDMA1_Channel1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(HPDMA1_Channel1_IRQn);
    
    /* HPDMA1_Channel2_IRQn interrupt configuration */
    HAL_NVIC_SetPriority(HPDMA1_Channel2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(HPDMA1_Channel2_IRQn);
}

/**
 * @brief SPI MSP Initialization
 * @param hspi: SPI handle pointer
 * @retval None
 */
void HAL_SPI_MspInit(SPI_HandleTypeDef* hspi)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    
    if (hspi->Instance == SPI1) {
        /* Peripheral clock enable */
        __HAL_RCC_SPI1_CLK_ENABLE();
        __HAL_RCC_GPIOA_CLK_ENABLE();
        
        /**SPI1 GPIO Configuration
        PA5     ------> SPI1_SCK
        PA6     ------> SPI1_MISO
        PA7     ------> SPI1_MOSI
        */
        GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
        HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
        
        /* SPI1 DMA Init */
        /* SPI1_RX Init */
        hdma_spi1_rx.Instance = HPDMA1_Channel0;
        hdma_spi1_rx.Init.Request = HPDMA1_REQUEST_SPI1_RX;
        hdma_spi1_rx.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
        hdma_spi1_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
        hdma_spi1_rx.Init.SrcInc = DMA_SINC_FIXED;
        hdma_spi1_rx.Init.DestInc = DMA_DINC_INCREMENTED;
        hdma_spi1_rx.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_BYTE;
        hdma_spi1_rx.Init.DestDataWidth = DMA_DEST_DATAWIDTH_BYTE;
        hdma_spi1_rx.Init.Priority = DMA_HIGH_PRIORITY;
        hdma_spi1_rx.Init.SrcBurstLength = 1;
        hdma_spi1_rx.Init.DestBurstLength = 1;
        hdma_spi1_rx.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT0 | DMA_DEST_ALLOCATED_PORT1;
        hdma_spi1_rx.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
        hdma_spi1_rx.Init.Mode = DMA_NORMAL;
        
        if (HAL_DMA_Init(&hdma_spi1_rx) != HAL_OK) {
            Error_Handler();
        }
        
        __HAL_LINKDMA(hspi, hdmarx, hdma_spi1_rx);
        
        /* SPI1 interrupt Init */
        HAL_NVIC_SetPriority(SPI1_IRQn, 0, 0);
        HAL_NVIC_EnableIRQ(SPI1_IRQn);
    }
    
    if (hspi->Instance == SPI2) {
        /* Peripheral clock enable */
        __HAL_RCC_SPI2_CLK_ENABLE();
        __HAL_RCC_GPIOB_CLK_ENABLE();
        
        /**SPI2 GPIO Configuration
        PB10     ------> SPI2_SCK
        PB14     ------> SPI2_MISO
        PB15     ------> SPI2_MOSI
        */
        GPIO_InitStruct.Pin = GPIO_PIN_10 | GPIO_PIN_14 | GPIO_PIN_15;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
        GPIO_InitStruct.Pull = GPIO_NOPULL;
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF5_SPI2;
        HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
        
        /* SPI2 DMA Init */
        /* SPI2_TX Init */
        hdma_spi2_tx.Instance = HPDMA1_Channel1;
        hdma_spi2_tx.Init.Request = HPDMA1_REQUEST_SPI2_TX;
        hdma_spi2_tx.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
        hdma_spi2_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
        hdma_spi2_tx.Init.SrcInc = DMA_SINC_INCREMENTED;
        hdma_spi2_tx.Init.DestInc = DMA_DINC_FIXED;
        hdma_spi2_tx.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_BYTE;
        hdma_spi2_tx.Init.DestDataWidth = DMA_DEST_DATAWIDTH_BYTE;
        hdma_spi2_tx.Init.Priority = DMA_HIGH_PRIORITY;
        hdma_spi2_tx.Init.SrcBurstLength = 1;
        hdma_spi2_tx.Init.DestBurstLength = 1;
        hdma_spi2_tx.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT0 | DMA_DEST_ALLOCATED_PORT1;
        hdma_spi2_tx.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
        hdma_spi2_tx.Init.Mode = DMA_NORMAL;
        
        if (HAL_DMA_Init(&hdma_spi2_tx) != HAL_OK) {
            Error_Handler();
        }
        
        __HAL_LINKDMA(hspi, hdmatx, hdma_spi2_tx);
        
        /* SPI2_RX Init */
        hdma_spi2_rx.Instance = HPDMA1_Channel2;
        hdma_spi2_rx.Init.Request = HPDMA1_REQUEST_SPI2_RX;
        hdma_spi2_rx.Init.BlkHWRequest = DMA_BREQ_SINGLE_BURST;
        hdma_spi2_rx.Init.Direction = DMA_PERIPH_TO_MEMORY;
        hdma_spi2_rx.Init.SrcInc = DMA_SINC_FIXED;
        hdma_spi2_rx.Init.DestInc = DMA_DINC_INCREMENTED;
        hdma_spi2_rx.Init.SrcDataWidth = DMA_SRC_DATAWIDTH_BYTE;
        hdma_spi2_rx.Init.DestDataWidth = DMA_DEST_DATAWIDTH_BYTE;
        hdma_spi2_rx.Init.Priority = DMA_HIGH_PRIORITY;
        hdma_spi2_rx.Init.SrcBurstLength = 1;
        hdma_spi2_rx.Init.DestBurstLength = 1;
        hdma_spi2_rx.Init.TransferAllocatedPort = DMA_SRC_ALLOCATED_PORT0 | DMA_DEST_ALLOCATED_PORT1;
        hdma_spi2_rx.Init.TransferEventMode = DMA_TCEM_BLOCK_TRANSFER;
        hdma_spi2_rx.Init.Mode = DMA_NORMAL;
        
        if (HAL_DMA_Init(&hdma_spi2_rx) != HAL_OK) {
            Error_Handler();
        }
        
        __HAL_LINKDMA(hspi, hdmarx, hdma_spi2_rx);
        
        /* SPI2 interrupt Init */
        HAL_NVIC_SetPriority(SPI2_IRQn, 1, 0);
        HAL_NVIC_EnableIRQ(SPI2_IRQn);
    }
}

/**
 * @brief SPI MSP De-Initialization
 * @param hspi: SPI handle pointer
 * @retval None
 */
void HAL_SPI_MspDeInit(SPI_HandleTypeDef* hspi)
{
    if (hspi->Instance == SPI1) {
        /* Peripheral clock disable */
        __HAL_RCC_SPI1_CLK_DISABLE();
        
        /**SPI1 GPIO Configuration
        PA5     ------> SPI1_SCK
        PA6     ------> SPI1_MISO
        PA7     ------> SPI1_MOSI
        */
        HAL_GPIO_DeInit(GPIOA, GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7);
        
        /* SPI1 DMA DeInit */
        HAL_DMA_DeInit(hspi->hdmarx);
        
        /* SPI1 interrupt DeInit */
        HAL_NVIC_DisableIRQ(SPI1_IRQn);
    }
    
    if (hspi->Instance == SPI2) {
        /* Peripheral clock disable */
        __HAL_RCC_SPI2_CLK_DISABLE();
        
        /**SPI2 GPIO Configuration
        PB10     ------> SPI2_SCK
        PB14     ------> SPI2_MISO
        PB15     ------> SPI2_MOSI
        */
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_10 | GPIO_PIN_14 | GPIO_PIN_15);
        
        /* SPI2 DMA DeInit */
        HAL_DMA_DeInit(hspi->hdmatx);
        HAL_DMA_DeInit(hspi->hdmarx);
        
        /* SPI2 interrupt DeInit */
        HAL_NVIC_DisableIRQ(SPI2_IRQn);
    }
}

/**
 * @brief TIM MSP Initialization
 * @param htim: TIM handle pointer
 * @retval None
 */
void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* htim)
{
    if (htim->Instance == TIM2) {
        /* Peripheral clock enable */
        __HAL_RCC_TIM2_CLK_ENABLE();
        
        /* TIM2 interrupt Init */
        HAL_NVIC_SetPriority(TIM2_IRQn, 5, 0);
        HAL_NVIC_EnableIRQ(TIM2_IRQn);
    }
}

/**
 * @brief TIM MSP De-Initialization
 * @param htim: TIM handle pointer
 * @retval None
 */
void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* htim)
{
    if (htim->Instance == TIM2) {
        /* Peripheral clock disable */
        __HAL_RCC_TIM2_CLK_DISABLE();
        
        /* TIM2 interrupt DeInit */
        HAL_NVIC_DisableIRQ(TIM2_IRQn);
    }
}

/**
 * @brief DMA conversion complete callback
 * @param hdma: DMA handle
 * @retval None
 */
void HAL_DMA_ConvCpltCallback(DMA_HandleTypeDef *hdma)
{
    extern SPI_HandleTypeDef hspi1;
    extern SPI_HandleTypeDef hspi2;
    
    if (hdma->Instance == HPDMA1_Channel0) {
        /* SPI1 RX DMA complete */
        HAL_SPI_DMAStop(&hspi1);
        extern ADS1299_HandleTypeDef hads1299;
        ADS1299_SPI_TxRxCpltCallback(&hads1299);
    }
    
    if (hdma->Instance == HPDMA1_Channel1) {
        /* SPI2 TX DMA complete */
        // Handle MAX78000 TX complete
    }
    
    if (hdma->Instance == HPDMA1_Channel2) {
        /* SPI2 RX DMA complete */
        // Handle MAX78000 RX complete
    }
}

/**
 * @brief DMA error callback
 * @param hdma: DMA handle
 * @retval None
 */
void HAL_DMA_ErrorCallback(DMA_HandleTypeDef *hdma)
{
    if (hdma->Instance == HPDMA1_Channel0) {
        /* SPI1 RX DMA error */
        extern ADS1299_HandleTypeDef hads1299;
        ADS1299_SPI_ErrorCallback(&hads1299);
    }
    
    if (hdma->Instance == HPDMA1_Channel1 || hdma->Instance == HPDMA1_Channel2) {
        /* SPI2 DMA error */
        extern MCU_CommHandleTypeDef hcomm;
        MCU_Comm_ErrorCallback(MCU_STATUS_ERROR);
    }
}

/**
 * @brief Interrupt handlers
 */
void HPDMA1_Channel0_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_spi1_rx);
}

void HPDMA1_Channel1_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_spi2_tx);
}

void HPDMA1_Channel2_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_spi2_rx);
}

void SPI1_IRQHandler(void)
{
    extern SPI_HandleTypeDef hspi1;
    HAL_SPI_IRQHandler(&hspi1);
}

void SPI2_IRQHandler(void)
{
    extern SPI_HandleTypeDef hspi2;
    HAL_SPI_IRQHandler(&hspi2);
}

void TIM2_IRQHandler(void)
{
    extern TIM_HandleTypeDef htim2;
    HAL_TIM_IRQHandler(&htim2);
}

void EXTI5_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_5);
}

void EXTI13_IRQHandler(void)
{
    HAL_GPIO_EXTI_IRQHandler(GPIO_PIN_13);
}