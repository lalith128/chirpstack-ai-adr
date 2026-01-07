# ChirpStack AI-ADR Architecture

## System Overview

The ChirpStack AI-ADR system consists of several interconnected components working together to provide intelligent, AI-driven Adaptive Data Rate optimization for LoRaWAN networks.

## Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     Physical Layer                            │
│  ┌────────────┐         ┌──────────┐        ┌──────────────┐ │
│  │ End Node   │◄───────►│ Gateway  │◄──────►│ ChirpStack   │ │
│  │ (Arduino)  │  LoRa   │          │  UDP   │ Server       │ │
│  └────────────┘         └──────────┘        └──────┬───────┘ │
└─────────────────────────────────────────────────────┼─────────┘
                                                      │
┌─────────────────────────────────────────────────────┼─────────┐
│                  Application Layer                  │         │
│                                                      ▼         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              MQTT Broker (mosquitto)                     │ │
│  │  Topics:                                                 │ │
│  │  • application/{id}/device/{eui}/event/up               │ │
│  │  • application/{id}/device/{eui}/command/down           │ │
│  └────────────┬────────────────────────────┬────────────────┘ │
│               │                            │                  │
│               ▼                            ▼                  │
│  ┌────────────────────┐      ┌────────────────────────────┐  │
│  │  JavaScript        │      │  Python ADR Engine         │  │
│  │  Decoder           │      │  ┌──────────────────────┐  │  │
│  │  (BME280)          │      │  │  Safety Shield       │  │  │
│  │                    │      │  │  • TFLite Model      │  │  │
│  │  Decodes:          │      │  │  • Scalers           │  │  │
│  │  • Temperature     │      │  │  • SNR Prediction    │  │  │
│  │  • Humidity        │      │  └──────────────────────┘  │  │
│  │  • Pressure        │      │  ┌──────────────────────┐  │  │
│  └────────────────────┘      │  │  Exploration Engine  │  │  │
│                              │  │  • Random DR/TX      │  │  │
│                              │  │  • Weighted Select   │  │  │
│                              │  └──────────────────────┘  │  │
│                              │  ┌──────────────────────┐  │  │
│                              │  │  MAC Command Gen     │  │  │
│                              │  │  • LinkADRReq        │  │  │
│                              │  └──────────────────────┘  │  │
│                              │  ┌──────────────────────┐  │  │
│                              │  │  CSV Logger          │  │  │
│                              │  │  • Transitions       │  │  │
│                              │  │  • Rewards           │  │  │
│                              │  └──────────────────────┘  │  │
│                              └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Uplink Flow

1. **End Node** reads BME280 sensor (temp, humidity, pressure)
2. **End Node** encodes data as 6-byte payload and transmits via LoRa
3. **Gateway** receives LoRa packet and forwards to ChirpStack via UDP
4. **ChirpStack** processes packet and runs JavaScript decoder
5. **JavaScript Decoder** unpacks payload into `{temp, hum, pres}`
6. **ChirpStack** publishes to MQTT topic `application/{id}/device/{eui}/event/up`
7. **ADR Engine** receives MQTT message with decoded data + RSSI/SNR

### ADR Decision Flow

8. **Exploration Engine** randomly selects DR/TX setting (weighted)
9. **Safety Shield** loads TFLite model and scalers
10. **Safety Shield** predicts SNR for proposed DR/TX
11. **Safety Shield** validates: `Predicted_SNR - 2σ >= Sensitivity + 5dB`
12. If **safe**: proceed with action
13. If **unsafe**: fallback to SF12/TX14
14. **MAC Command Generator** creates `LinkADRReq` bytes
15. **CSV Logger** calculates reward and logs transition

### Downlink Flow

16. **ADR Engine** publishes MAC command to MQTT topic `application/{id}/device/{eui}/command/down`
17. **ChirpStack** receives downlink command
18. **ChirpStack** encrypts MAC command and schedules transmission
19. **Gateway** transmits downlink in RX1 or RX2 window
20. **End Node** receives and applies new DR/TX settings automatically

## Key Components

### 1. End Node (Arduino + LMIC)

**Hardware**:
- Arduino board (e.g., Arduino Uno, Mega)
- LoRa module (e.g., RFM95W)
- BME280 sensor (I2C)

**Software**:
- MCCI LoRaWAN LMIC library
- Adafruit BME280 library

**Responsibilities**:
- Read sensor data
- Encode payload (6 bytes)
- Transmit via LoRa
- Receive and apply MAC commands

### 2. ChirpStack Server

**Components**:
- Network Server
- Application Server
- JavaScript Codec Engine
- MQTT Integration

**Responsibilities**:
- Manage device sessions
- Decode/encode payloads
- Publish uplinks to MQTT
- Subscribe to downlinks from MQTT
- Handle MAC command encryption

### 3. MQTT Broker (mosquitto)

**Topics**:
- `application/{app_id}/device/{dev_eui}/event/up` - Uplink events
- `application/{app_id}/device/{dev_eui}/command/down` - Downlink commands

**Responsibilities**:
- Message routing between ChirpStack and ADR Engine
- QoS handling
- Persistent connections

### 4. Python ADR Engine

**Modules**:

#### Safety Shield
- **Input**: Current state (RSSI, SNR, temp, hum, pres) + Proposed action (DR, TX)
- **Process**: 
  1. Scale context features
  2. Run TFLite inference
  3. Predict SNR with uncertainty
  4. Validate against sensitivity threshold
- **Output**: Boolean (safe/unsafe) + Predicted SNR

#### Exploration Engine
- **Strategy**: Weighted random selection
- **DR Weights**: SF7=30%, SF8=25%, SF9=20%, SF10=15%, SF11=5%, SF12=5%
- **TX Range**: 2-14 dBm (uniform random)

#### MAC Command Generator
- **Format**: `[CID, DR_TX, ChMask_LSB, ChMask_MSB, Redundancy]`
- **CID**: 0x03 (LinkADRReq)
- **DR_TX**: Combined data rate and TX power byte
- **ChMask**: Channel mask (0xFFFF for all channels)
- **Redundancy**: NbTrans and ChMaskCntl

#### CSV Logger
- **Reward Function**: `0.5 * (dr/5) + 0.5 * ((14-tx)/12)`
- **Columns**: rssi, snr, temp, hum, pres, dr, tx, reward
- **Format**: CSV with header

## Safety Shield Algorithm

```python
def is_safe_action(rssi, snr, temp, hum, pres, dr, tx):
    # 1. Predict SNR
    predicted_snr, sigma = model.predict(context, action)
    
    # 2. Get sensitivity threshold
    sf = DR_TO_SF[dr]
    sensitivity = SENSITIVITY[sf]
    
    # 3. Calculate lower bound
    lower_bound_snr = predicted_snr - 2 * sigma
    
    # 4. Validate
    is_safe = (rssi + lower_bound_snr) >= (sensitivity + 5)
    
    return is_safe, predicted_snr
```

## Deployment Architecture

```
┌─────────────────────────────────────────┐
│  Docker Host (Raspberry Pi / Linux)    │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  chirpstack-docker_chirpstack_1   │  │
│  │  Port: 8080                       │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  chirpstack-docker_mosquitto_1    │  │
│  │  Port: 1883                       │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  chirpstack-docker_postgres_1     │  │
│  │  Port: 5432 (internal)            │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  chirpstack-docker_redis_1        │  │
│  │  Port: 6379 (internal)            │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  chirpstack-docker_adr-engine_1   │  │
│  │  Volumes:                         │  │
│  │  • ./data:/app/data               │  │
│  │  • ./models:/app/models           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Network Topology

```
Internet
    │
    ▼
┌─────────────┐
│ ChirpStack  │ :8080 (Web UI)
│ Server      │
└──────┬──────┘
       │
       ├──────► PostgreSQL (Device DB)
       ├──────► Redis (Cache)
       └──────► MQTT Broker :1883
                    │
                    ├──────► ADR Engine (Subscribe: up)
                    └──────◄ ADR Engine (Publish: down)
```

## Security Considerations

1. **MQTT**: Currently no authentication (localhost only)
2. **ChirpStack UI**: Default admin credentials should be changed
3. **MAC Commands**: Encrypted by ChirpStack using device session keys
4. **Data Privacy**: CSV logs contain sensor data - ensure proper access control

## Performance Characteristics

- **Latency**: ~100-500ms from uplink to downlink
- **Throughput**: Handles 100+ devices per gateway
- **Storage**: ~1KB per transition (CSV)
- **Memory**: ~2GB for ADR engine container (TensorFlow)
- **CPU**: Minimal (<5% on Raspberry Pi 5)

## Future Enhancements

1. **Multi-Gateway Support**: Aggregate RSSI/SNR from multiple gateways
2. **Device Clustering**: Group devices by location/behavior
3. **Online Learning**: Update model based on real-world performance
4. **Dashboard**: Real-time visualization of DR/TX distributions
5. **A/B Testing**: Compare AI-ADR vs native ADR performance
