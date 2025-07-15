/**
 * @file mcu_communication.h
 * @brief STM32 - MAX78000 Communication Protocol Header
 * @author Dual-MCU EMG System
 * @date 2025
 */

#ifndef MCU_COMMUNICATION_H
#define MCU_COMMUNICATION_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include <stdint.h>
#include <stdbool.h>

/* Protocol Configuration */
#define MCU_COMM_PROTOCOL_VERSION   0x01
#define MCU_COMM_MAX_PAYLOAD_SIZE   128
#define MCU_COMM_HEADER_SIZE        8
#define MCU_COMM_CHECKSUM_SIZE      4
#define MCU_COMM_MAX_PACKET_SIZE    (MCU_COMM_HEADER_SIZE + MCU_COMM_MAX_PAYLOAD_SIZE + MCU_COMM_CHECKSUM_SIZE)

/* Communication Parameters */
#define MCU_COMM_TIMEOUT_MS         100
#define MCU_COMM_RETRY_COUNT        3
#define MCU_COMM_ACK_TIMEOUT_MS     50
#define MCU_COMM_HEARTBEAT_INTERVAL 1000    // 1 second

/* Packet Types */
typedef enum {
    MCU_PACKET_FEATURE_DATA = 0x01,         // Feature vector data
    MCU_PACKET_CLASSIFICATION = 0x02,       // Classification result
    MCU_PACKET_CONTROL = 0x03,              // Control commands
    MCU_PACKET_STATUS = 0x04,               // Status information
    MCU_PACKET_CONFIG = 0x05,               // Configuration data
    MCU_PACKET_HEARTBEAT = 0x06,            // Heartbeat packet
    MCU_PACKET_ACK = 0x07,                  // Acknowledgment
    MCU_PACKET_NACK = 0x08,                 // Negative acknowledgment
    MCU_PACKET_ERROR = 0x09                 // Error packet
} MCU_PacketTypeTypeDef;

/* Control Commands */
typedef enum {
    MCU_CMD_START_INFERENCE = 0x01,         // Start inference
    MCU_CMD_STOP_INFERENCE = 0x02,          // Stop inference
    MCU_CMD_RESET_SYSTEM = 0x03,            // Reset system
    MCU_CMD_CALIBRATE = 0x04,               // Calibrate system
    MCU_CMD_UPDATE_CONFIG = 0x05,           // Update configuration
    MCU_CMD_REQUEST_STATUS = 0x06,          // Request status
    MCU_CMD_LOAD_MODEL = 0x07,              // Load new model
    MCU_CMD_SAVE_DATA = 0x08                // Save data request
} MCU_CommandTypeDef;

/* Status Codes */
typedef enum {
    MCU_STATUS_OK = 0x00,                   // Success
    MCU_STATUS_BUSY = 0x01,                 // System busy
    MCU_STATUS_ERROR = 0x02,                // General error
    MCU_STATUS_TIMEOUT = 0x03,              // Timeout error
    MCU_STATUS_INVALID_PACKET = 0x04,       // Invalid packet
    MCU_STATUS_CHECKSUM_ERROR = 0x05,       // Checksum error
    MCU_STATUS_BUFFER_FULL = 0x06,          // Buffer full
    MCU_STATUS_NOT_INITIALIZED = 0x07,      // Not initialized
    MCU_STATUS_INVALID_COMMAND = 0x08,      // Invalid command
    MCU_STATUS_INFERENCE_ERROR = 0x09       // Inference error
} MCU_StatusTypeDef;

/* Packet Header Structure */
typedef struct {
    uint8_t sync_byte;                      // Synchronization byte (0xAA)
    uint8_t version;                        // Protocol version
    uint8_t packet_type;                    // Packet type
    uint8_t sequence_number;                // Sequence number
    uint16_t payload_length;                // Payload length
    uint8_t flags;                          // Packet flags
    uint8_t reserved;                       // Reserved byte
} __attribute__((packed)) MCU_PacketHeaderTypeDef;

/* Feature Data Packet */
typedef struct {
    uint32_t timestamp;                     // Timestamp (microseconds)
    uint8_t channel_count;                  // Number of channels
    uint8_t feature_count;                  // Features per channel
    uint8_t data_format;                    // Data format (0=float, 1=int8)
    uint8_t reserved;                       // Reserved
    int8_t features[72];                    // Feature data (8 channels × 9 features)
} __attribute__((packed)) MCU_FeatureDataTypeDef;

/* Classification Result Packet */
typedef struct {
    uint32_t timestamp;                     // Timestamp (microseconds)
    uint8_t class_count;                    // Number of classes
    uint8_t predicted_class;                // Predicted class index
    uint16_t confidence;                    // Confidence score (0-1000)
    uint8_t class_scores[8];                // Individual class scores
    uint8_t processing_time;                // Processing time (ms)
    uint8_t model_version;                  // Model version
    uint8_t reserved;                       // Reserved
} __attribute__((packed)) MCU_ClassificationTypeDef;

/* Control Packet */
typedef struct {
    uint8_t command;                        // Command type
    uint8_t parameter_count;                // Number of parameters
    uint16_t reserved;                      // Reserved
    uint32_t parameters[8];                 // Command parameters
} __attribute__((packed)) MCU_ControlTypeDef;

/* Status Packet */
typedef struct {
    uint8_t system_status;                  // System status
    uint8_t stm32_status;                   // STM32 status
    uint8_t max78000_status;                // MAX78000 status
    uint8_t error_code;                     // Last error code
    uint32_t uptime;                        // System uptime (seconds)
    uint32_t processed_samples;             // Processed samples count
    uint32_t inference_count;               // Inference count
    uint16_t cpu_usage;                     // CPU usage (0-1000)
    uint16_t memory_usage;                  // Memory usage (0-1000)
} __attribute__((packed)) MCU_StatusPacketTypeDef;

/* Complete Packet Structure */
typedef struct {
    MCU_PacketHeaderTypeDef header;         // Packet header
    union {
        MCU_FeatureDataTypeDef feature_data;
        MCU_ClassificationTypeDef classification;
        MCU_ControlTypeDef control;
        MCU_StatusPacketTypeDef status;
        uint8_t raw_data[MCU_COMM_MAX_PAYLOAD_SIZE];
    } payload;
    uint32_t checksum;                      // CRC32 checksum
} __attribute__((packed)) MCU_PacketTypeDef;

/* Communication Statistics */
typedef struct {
    uint32_t packets_sent;                  // Packets sent
    uint32_t packets_received;              // Packets received
    uint32_t packets_lost;                  // Packets lost
    uint32_t checksum_errors;               // Checksum errors
    uint32_t timeout_errors;                // Timeout errors
    uint32_t last_heartbeat;                // Last heartbeat timestamp
    uint16_t round_trip_time;               // Average round trip time (ms)
    uint8_t connection_quality;             // Connection quality (0-100)
} MCU_CommStatsTypeDef;

/* Communication Handle */
typedef struct {
    void *hw_handle;                        // Hardware handle (SPI/UART)
    uint8_t *tx_buffer;                     // Transmission buffer
    uint8_t *rx_buffer;                     // Reception buffer
    uint16_t tx_buffer_size;                // TX buffer size
    uint16_t rx_buffer_size;                // RX buffer size
    
    uint8_t current_seq_num;                // Current sequence number
    uint8_t expected_seq_num;               // Expected sequence number
    
    MCU_CommStatsTypeDef stats;             // Communication statistics
    
    bool is_initialized;                    // Initialization flag
    bool is_connected;                      // Connection status
    
    void (*packet_received_callback)(MCU_PacketTypeDef *packet);
    void (*error_callback)(MCU_StatusTypeDef error);
    void (*connection_status_callback)(bool connected);
} MCU_CommHandleTypeDef;

/* Function Prototypes */

/* Initialization and Configuration */
MCU_StatusTypeDef MCU_Comm_Init(MCU_CommHandleTypeDef *hcomm, void *hw_handle);
MCU_StatusTypeDef MCU_Comm_DeInit(MCU_CommHandleTypeDef *hcomm);
MCU_StatusTypeDef MCU_Comm_Start(MCU_CommHandleTypeDef *hcomm);
MCU_StatusTypeDef MCU_Comm_Stop(MCU_CommHandleTypeDef *hcomm);

/* Packet Functions */
MCU_StatusTypeDef MCU_Comm_SendPacket(MCU_CommHandleTypeDef *hcomm, MCU_PacketTypeDef *packet);
MCU_StatusTypeDef MCU_Comm_ReceivePacket(MCU_CommHandleTypeDef *hcomm, MCU_PacketTypeDef *packet);
MCU_StatusTypeDef MCU_Comm_ProcessReceivedData(MCU_CommHandleTypeDef *hcomm, uint8_t *data, uint16_t length);

/* High-Level Communication Functions */
MCU_StatusTypeDef MCU_Comm_SendFeatureData(MCU_CommHandleTypeDef *hcomm, MCU_FeatureDataTypeDef *features);
MCU_StatusTypeDef MCU_Comm_SendClassification(MCU_CommHandleTypeDef *hcomm, MCU_ClassificationTypeDef *classification);
MCU_StatusTypeDef MCU_Comm_SendControl(MCU_CommHandleTypeDef *hcomm, MCU_ControlTypeDef *control);
MCU_StatusTypeDef MCU_Comm_SendStatus(MCU_CommHandleTypeDef *hcomm, MCU_StatusPacketTypeDef *status);
MCU_StatusTypeDef MCU_Comm_SendHeartbeat(MCU_CommHandleTypeDef *hcomm);

/* Utility Functions */
MCU_StatusTypeDef MCU_Comm_CreatePacket(MCU_PacketTypeDef *packet, MCU_PacketTypeTypeDef type, 
                                        void *payload, uint16_t payload_size);
MCU_StatusTypeDef MCU_Comm_ValidatePacket(MCU_PacketTypeDef *packet);
uint32_t MCU_Comm_CalculateChecksum(uint8_t *data, uint16_t length);
bool MCU_Comm_IsPacketValid(MCU_PacketTypeDef *packet);

/* Statistics Functions */
void MCU_Comm_UpdateStats(MCU_CommHandleTypeDef *hcomm, bool packet_sent, bool packet_received);
void MCU_Comm_GetStats(MCU_CommHandleTypeDef *hcomm, MCU_CommStatsTypeDef *stats);
void MCU_Comm_ResetStats(MCU_CommHandleTypeDef *hcomm);

/* Callback Functions */
void MCU_Comm_PacketReceivedCallback(MCU_PacketTypeDef *packet);
void MCU_Comm_ErrorCallback(MCU_StatusTypeDef error);
void MCU_Comm_ConnectionStatusCallback(bool connected);

/* Hardware Abstraction Layer */
MCU_StatusTypeDef MCU_Comm_HAL_Init(void *hw_handle);
MCU_StatusTypeDef MCU_Comm_HAL_Transmit(void *hw_handle, uint8_t *data, uint16_t length);
MCU_StatusTypeDef MCU_Comm_HAL_Receive(void *hw_handle, uint8_t *data, uint16_t length);
MCU_StatusTypeDef MCU_Comm_HAL_TransmitReceive(void *hw_handle, uint8_t *tx_data, uint8_t *rx_data, uint16_t length);

/* Constants */
#define MCU_COMM_SYNC_BYTE          0xAA
#define MCU_COMM_FLAG_ACK_REQ       0x01
#define MCU_COMM_FLAG_PRIORITY      0x02
#define MCU_COMM_FLAG_COMPRESSED    0x04
#define MCU_COMM_FLAG_ENCRYPTED     0x08

#ifdef __cplusplus
}
#endif

#endif /* MCU_COMMUNICATION_H */