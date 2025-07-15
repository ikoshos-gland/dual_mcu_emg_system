/**
 * @file emg_communication.c
 * @brief MAX78000 Communication Interface Implementation
 * @author Dual-MCU EMG System
 * @date 2025
 */

/* Includes ------------------------------------------------------------------*/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc_device.h"
#include "mxc_delay.h"
#include "spi.h"
#include "gpio.h"
#include "dma.h"
#include "emg_communication.h"
#include "mcu_communication.h"

/* Private defines -----------------------------------------------------------*/
#define COMM_BUFFER_SIZE            256
#define COMM_TIMEOUT_MS             100
#define COMM_MAX_RETRIES            3
#define COMM_HEARTBEAT_INTERVAL     1000    // 1 second

/* Private variables ---------------------------------------------------------*/
static mxc_spi_regs_t *spi_instance = NULL;
static mxc_gpio_regs_t *cs_port = NULL;
static mxc_gpio_regs_t *irq_port = NULL;
static uint32_t cs_pin = 0;
static uint32_t irq_pin = 0;

static uint8_t tx_buffer[COMM_BUFFER_SIZE];
static uint8_t rx_buffer[COMM_BUFFER_SIZE];
static volatile bool spi_complete = false;
static volatile bool irq_received = false;

static EMG_Communication_FeatureCallback feature_callback = NULL;
static EMG_Communication_ErrorCallback error_callback = NULL;

static uint32_t last_heartbeat = 0;
static uint32_t packets_received = 0;
static uint32_t packets_sent = 0;
static uint32_t communication_errors = 0;

/* Private function prototypes -----------------------------------------------*/
static EMG_Communication_Status SPI_Transmit(const uint8_t *data, uint16_t size);
static EMG_Communication_Status SPI_Receive(uint8_t *data, uint16_t size);
static EMG_Communication_Status SPI_TransmitReceive(const uint8_t *tx_data, uint8_t *rx_data, uint16_t size);
static void CS_Enable(void);
static void CS_Disable(void);
static bool Is_IRQ_Active(void);
static void Process_Received_Packet(const uint8_t *data, uint16_t size);
static EMG_Communication_Status Send_Packet(const MCU_PacketTypeDef *packet);
static EMG_Communication_Status Receive_Packet(MCU_PacketTypeDef *packet);
static void SPI_Callback(mxc_spi_req_t *req, int error);
static void IRQ_Callback(void *cbdata);

/* Public functions ----------------------------------------------------------*/

/**
 * @brief Initialize communication interface
 * @param config: Configuration structure
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_Init(const EMG_Communication_Config *config)
{
    if (config == NULL) {
        return EMG_COMM_ERROR_INVALID_PARAM;
    }
    
    /* Store configuration */
    spi_instance = config->spi_instance;
    cs_port = config->cs_port;
    cs_pin = config->cs_pin;
    irq_port = config->irq_port;
    irq_pin = config->irq_pin;
    feature_callback = config->feature_callback;
    error_callback = config->error_callback;
    
    /* Initialize SPI */
    mxc_spi_init_t spi_init = {
        .type = MXC_SPI_TYPE_SLAVE,
        .mode = MXC_SPI_MODE_0,
        .freq = config->baudrate,
        .ss_pol = MXC_SPI_POL_LOW,
        .use_dma = true,
        .dma_tx_ch = 0,
        .dma_rx_ch = 1,
        .callback = SPI_Callback
    };
    
    if (MXC_SPI_Init(spi_instance, &spi_init) != E_NO_ERROR) {
        return EMG_COMM_ERROR_SPI_INIT;
    }
    
    /* Configure CS pin as input (controlled by master) */
    mxc_gpio_cfg_t cs_cfg = {
        .port = cs_port,
        .mask = (1 << cs_pin),
        .func = MXC_GPIO_FUNC_IN,
        .pad = MXC_GPIO_PAD_PULL_UP
    };
    MXC_GPIO_Config(&cs_cfg);
    
    /* Configure IRQ pin as output */
    mxc_gpio_cfg_t irq_cfg = {
        .port = irq_port,
        .mask = (1 << irq_pin),
        .func = MXC_GPIO_FUNC_OUT,
        .pad = MXC_GPIO_PAD_NONE
    };
    MXC_GPIO_Config(&irq_cfg);
    MXC_GPIO_OutClr(irq_port, (1 << irq_pin)); // Start low
    
    /* Enable SPI interrupt */
    NVIC_EnableIRQ(MXC_SPI_GET_IRQ(MXC_SPI_GET_IDX(spi_instance)));
    
    /* Reset statistics */
    packets_received = 0;
    packets_sent = 0;
    communication_errors = 0;
    last_heartbeat = MXC_GetTickCount();
    
    return EMG_COMM_OK;
}

/**
 * @brief De-initialize communication interface
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_DeInit(void)
{
    if (spi_instance == NULL) {
        return EMG_COMM_ERROR_NOT_INITIALIZED;
    }
    
    /* Disable SPI interrupt */
    NVIC_DisableIRQ(MXC_SPI_GET_IRQ(MXC_SPI_GET_IDX(spi_instance)));
    
    /* Shutdown SPI */
    MXC_SPI_Shutdown(spi_instance);
    
    /* Reset variables */
    spi_instance = NULL;
    cs_port = NULL;
    irq_port = NULL;
    feature_callback = NULL;
    error_callback = NULL;
    
    return EMG_COMM_OK;
}

/**
 * @brief Process communication events
 * @retval None
 */
void EMG_Communication_Process(void)
{
    MCU_PacketTypeDef packet;
    
    /* Check for incoming data */
    if (spi_complete) {
        spi_complete = false;
        Process_Received_Packet(rx_buffer, sizeof(rx_buffer));
    }
    
    /* Send periodic heartbeat */
    uint32_t current_time = MXC_GetTickCount();
    if (current_time - last_heartbeat >= COMM_HEARTBEAT_INTERVAL) {
        EMG_Communication_SendHeartbeat();
        last_heartbeat = current_time;
    }
}

/**
 * @brief Send classification result
 * @param result: Classification result
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_SendClassification(const EMG_CNN_ResultTypeDef *result)
{
    if (result == NULL) {
        return EMG_COMM_ERROR_INVALID_PARAM;
    }
    
    MCU_PacketTypeDef packet;
    MCU_ClassificationTypeDef *classification = &packet.payload.classification;
    
    /* Create packet header */
    packet.header.sync_byte = MCU_COMM_SYNC_BYTE;
    packet.header.version = MCU_COMM_PROTOCOL_VERSION;
    packet.header.packet_type = MCU_PACKET_CLASSIFICATION;
    packet.header.sequence_number = 0; // Would be managed by protocol layer
    packet.header.payload_length = sizeof(MCU_ClassificationTypeDef);
    packet.header.flags = 0;
    packet.header.reserved = 0;
    
    /* Fill classification data */
    classification->timestamp = result->timestamp;
    classification->class_count = EMG_CNN_OUTPUT_SIZE;
    classification->predicted_class = result->predicted_class;
    classification->confidence = result->confidence;
    
    /* Copy class scores */
    for (int i = 0; i < EMG_CNN_OUTPUT_SIZE && i < 8; i++) {
        classification->class_scores[i] = (uint8_t)(result->class_scores[i] / 10); // Scale to 0-100
    }
    
    classification->processing_time = (uint8_t)(result->inference_time / 1000); // Convert to ms
    classification->model_version = 1;
    classification->reserved = 0;
    
    /* Calculate checksum */
    packet.checksum = MCU_Comm_CalculateChecksum((uint8_t *)&packet, 
                                                sizeof(packet) - sizeof(packet.checksum));
    
    /* Send packet */
    return Send_Packet(&packet);
}

/**
 * @brief Send heartbeat packet
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_SendHeartbeat(void)
{
    MCU_PacketTypeDef packet;
    
    /* Create heartbeat packet */
    packet.header.sync_byte = MCU_COMM_SYNC_BYTE;
    packet.header.version = MCU_COMM_PROTOCOL_VERSION;
    packet.header.packet_type = MCU_PACKET_HEARTBEAT;
    packet.header.sequence_number = 0;
    packet.header.payload_length = 0;
    packet.header.flags = 0;
    packet.header.reserved = 0;
    
    /* Calculate checksum */
    packet.checksum = MCU_Comm_CalculateChecksum((uint8_t *)&packet, 
                                                sizeof(packet.header) + packet.header.payload_length);
    
    /* Send packet */
    return Send_Packet(&packet);
}

/**
 * @brief Send status packet
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_SendStatus(void)
{
    MCU_PacketTypeDef packet;
    MCU_StatusPacketTypeDef *status = &packet.payload.status;
    
    /* Create status packet */
    packet.header.sync_byte = MCU_COMM_SYNC_BYTE;
    packet.header.version = MCU_COMM_PROTOCOL_VERSION;
    packet.header.packet_type = MCU_PACKET_STATUS;
    packet.header.sequence_number = 0;
    packet.header.payload_length = sizeof(MCU_StatusPacketTypeDef);
    packet.header.flags = 0;
    packet.header.reserved = 0;
    
    /* Fill status data */
    status->system_status = MCU_STATUS_OK;
    status->stm32_status = MCU_STATUS_OK;
    status->max78000_status = MCU_STATUS_OK;
    status->error_code = 0;
    status->uptime = MXC_GetTickCount() / 1000; // Convert to seconds
    status->processed_samples = 0; // Would be tracked
    status->inference_count = packets_received;
    status->cpu_usage = 500; // 50% (would be measured)
    status->memory_usage = 300; // 30% (would be measured)
    
    /* Calculate checksum */
    packet.checksum = MCU_Comm_CalculateChecksum((uint8_t *)&packet, 
                                                sizeof(packet.header) + packet.header.payload_length);
    
    /* Send packet */
    return Send_Packet(&packet);
}

/**
 * @brief Reset communication interface
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_Reset(void)
{
    /* Reset SPI */
    MXC_SPI_AbortAsync(spi_instance);
    MXC_SPI_ClearFlags(spi_instance, 0xFFFFFFFF);
    
    /* Reset state variables */
    spi_complete = false;
    irq_received = false;
    
    /* Reset statistics */
    communication_errors = 0;
    
    return EMG_COMM_OK;
}

/**
 * @brief Get communication statistics
 * @param stats: Pointer to statistics structure
 * @retval EMG_Communication_Status
 */
EMG_Communication_Status EMG_Communication_GetStats(EMG_Communication_Stats *stats)
{
    if (stats == NULL) {
        return EMG_COMM_ERROR_INVALID_PARAM;
    }
    
    stats->packets_received = packets_received;
    stats->packets_sent = packets_sent;
    stats->communication_errors = communication_errors;
    stats->uptime = MXC_GetTickCount() / 1000;
    
    return EMG_COMM_OK;
}

/* Private functions ---------------------------------------------------------*/

/**
 * @brief SPI transmit function
 * @param data: Data to transmit
 * @param size: Data size
 * @retval EMG_Communication_Status
 */
static EMG_Communication_Status SPI_Transmit(const uint8_t *data, uint16_t size)
{
    mxc_spi_req_t req = {
        .spi = spi_instance,
        .txData = (uint8_t *)data,
        .rxData = NULL,
        .txLen = size,
        .rxLen = 0,
        .ssIdx = 0,
        .ssDeassert = 1,
        .txCnt = 0,
        .rxCnt = 0,
        .completeCB = SPI_Callback
    };
    
    spi_complete = false;
    
    if (MXC_SPI_MasterTransactionAsync(&req) != E_NO_ERROR) {
        return EMG_COMM_ERROR_SPI_TRANSMIT;
    }
    
    /* Wait for completion */
    uint32_t timeout = MXC_GetTickCount() + COMM_TIMEOUT_MS;
    while (!spi_complete && MXC_GetTickCount() < timeout) {
        // Wait
    }
    
    if (!spi_complete) {
        return EMG_COMM_ERROR_TIMEOUT;
    }
    
    return EMG_COMM_OK;
}

/**
 * @brief SPI receive function
 * @param data: Buffer to receive data
 * @param size: Buffer size
 * @retval EMG_Communication_Status
 */
static EMG_Communication_Status SPI_Receive(uint8_t *data, uint16_t size)
{
    mxc_spi_req_t req = {
        .spi = spi_instance,
        .txData = NULL,
        .rxData = data,
        .txLen = 0,
        .rxLen = size,
        .ssIdx = 0,
        .ssDeassert = 1,
        .txCnt = 0,
        .rxCnt = 0,
        .completeCB = SPI_Callback
    };
    
    spi_complete = false;
    
    if (MXC_SPI_SlaveTransactionAsync(&req) != E_NO_ERROR) {
        return EMG_COMM_ERROR_SPI_RECEIVE;
    }
    
    /* Wait for completion */
    uint32_t timeout = MXC_GetTickCount() + COMM_TIMEOUT_MS;
    while (!spi_complete && MXC_GetTickCount() < timeout) {
        // Wait
    }
    
    if (!spi_complete) {
        return EMG_COMM_ERROR_TIMEOUT;
    }
    
    return EMG_COMM_OK;
}

/**
 * @brief Process received packet
 * @param data: Received data
 * @param size: Data size
 * @retval None
 */
static void Process_Received_Packet(const uint8_t *data, uint16_t size)
{
    MCU_PacketTypeDef *packet = (MCU_PacketTypeDef *)data;
    
    /* Validate packet */
    if (packet->header.sync_byte != MCU_COMM_SYNC_BYTE) {
        communication_errors++;
        if (error_callback) {
            error_callback(EMG_COMM_ERROR_INVALID_SYNC);
        }
        return;
    }
    
    /* Verify checksum */
    uint32_t calculated_checksum = MCU_Comm_CalculateChecksum((uint8_t *)packet,
                                                             sizeof(packet->header) + packet->header.payload_length);
    if (calculated_checksum != packet->checksum) {
        communication_errors++;
        if (error_callback) {
            error_callback(EMG_COMM_ERROR_CHECKSUM);
        }
        return;
    }
    
    packets_received++;
    
    /* Process packet based on type */
    switch (packet->header.packet_type) {
        case MCU_PACKET_FEATURE_DATA:
            if (feature_callback) {
                MCU_FeatureDataTypeDef *feature_data = &packet->payload.feature_data;
                feature_callback(feature_data->features, sizeof(feature_data->features));
            }
            break;
            
        case MCU_PACKET_CONTROL:
            /* Handle control commands */
            break;
            
        case MCU_PACKET_HEARTBEAT:
            /* Heartbeat received - update connection status */
            break;
            
        default:
            communication_errors++;
            if (error_callback) {
                error_callback(EMG_COMM_ERROR_UNKNOWN_PACKET);
            }
            break;
    }
}

/**
 * @brief Send packet
 * @param packet: Packet to send
 * @retval EMG_Communication_Status
 */
static EMG_Communication_Status Send_Packet(const MCU_PacketTypeDef *packet)
{
    /* Signal IRQ to master */
    MXC_GPIO_OutSet(irq_port, (1 << irq_pin));
    
    /* Wait for master to assert CS */
    uint32_t timeout = MXC_GetTickCount() + COMM_TIMEOUT_MS;
    while (MXC_GPIO_InGet(cs_port, (1 << cs_pin)) && MXC_GetTickCount() < timeout) {
        // Wait for CS assertion
    }
    
    if (MXC_GPIO_InGet(cs_port, (1 << cs_pin))) {
        MXC_GPIO_OutClr(irq_port, (1 << irq_pin));
        return EMG_COMM_ERROR_TIMEOUT;
    }
    
    /* Transmit packet */
    EMG_Communication_Status status = SPI_Transmit((uint8_t *)packet, 
                                                  sizeof(packet->header) + packet->header.payload_length + sizeof(packet->checksum));
    
    /* Clear IRQ */
    MXC_GPIO_OutClr(irq_port, (1 << irq_pin));
    
    if (status == EMG_COMM_OK) {
        packets_sent++;
    } else {
        communication_errors++;
    }
    
    return status;
}

/**
 * @brief SPI callback function
 * @param req: SPI request
 * @param error: Error code
 * @retval None
 */
static void SPI_Callback(mxc_spi_req_t *req, int error)
{
    if (error == E_NO_ERROR) {
        spi_complete = true;
    } else {
        communication_errors++;
        if (error_callback) {
            error_callback(EMG_COMM_ERROR_SPI_CALLBACK);
        }
    }
}

/**
 * @brief IRQ callback function
 * @param cbdata: Callback data
 * @retval None
 */
static void IRQ_Callback(void *cbdata)
{
    irq_received = true;
    
    /* Start SPI reception */
    SPI_Receive(rx_buffer, sizeof(rx_buffer));
}