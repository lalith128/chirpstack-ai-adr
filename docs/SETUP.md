# Phase 1 AI-Driven LoRaWAN Data Collector - Setup Guide

## Overview

This system implements an external ADR engine that uses an AI "Safety Shield" to explore different Data Rate (DR) and Transmit Power (TX) settings while logging results to CSV for future Reinforcement Learning.

## Architecture

```
End Node (Arduino) → Gateway → ChirpStack → MQTT → ADR Engine (Python)
                                                        ↓
                                                  CSV Logger
```

## Files Created

- **`codec_bme280.js`**: JavaScript decoder for ChirpStack Device Profile
- **`adr_engine_phase1.py`**: Python ADR engine with Safety Shield
- **`requirements.txt`**: Python dependencies
- **`Dockerfile.adr`**: Docker image for ADR engine
- **`docker-compose.yml`**: Updated with `adr-engine` service

## Setup Instructions

### 1. Configure ChirpStack Device Profile

1. Log into ChirpStack UI at `http://localhost:8080`
2. Navigate to **Device Profiles** → Select/Create your device profile
3. Under **Codec** section:
   - Set **Payload codec** to "JavaScript"
   - Copy the contents of `codec_bme280.js` into the **Decoder** field
4. Under **ADR** section:
   - **CRITICAL**: Set **ADR algorithm** to "Disabled" OR uncheck "ADR enabled"
   - This prevents conflicts with the external ADR engine
5. Save the device profile

### 2. Verify Region Configuration (Optional)

If you want to disable ADR globally for the region:

Edit `configuration/chirpstack/region_eu868.toml` (or your region file):
```toml
[regions.network]
  adr_disabled=true  # Change from false to true
```

### 3. Build and Start the Stack

```bash
cd /mnt/c/Users/LALITH\ V/Desktop/doc/chirpstack-docker
docker-compose up -d --build
```

This will:
- Build the ADR engine Docker image
- Start all services (ChirpStack, MQTT, Postgres, Redis, ADR Engine)

### 4. Verify Services are Running

```bash
docker-compose ps
```

You should see all services in "Up" state, including `adr-engine`.

### 5. Monitor ADR Engine Logs

```bash
docker-compose logs -f adr-engine
```

You should see:
- "Loading TFLite model and scalers..."
- "Connected to MQTT broker"
- "Subscribed to application/+/device/+/event/up"

## How It Works

### 1. Uplink Processing

When your Arduino node sends sensor data:

1. **ChirpStack** receives the LoRa packet
2. **JavaScript Decoder** unpacks the 6-byte payload into `{temp, hum, pres}`
3. **MQTT** publishes the uplink event to `application/{app_id}/device/{dev_eui}/event/up`

### 2. ADR Decision

The Python ADR engine:

1. **Receives** the uplink JSON via MQTT
2. **Explores** by randomly selecting a DR/TX setting (favoring higher DR, lower TX)
3. **Safety Shield** predicts SNR using the TFLite model
4. **Validates** if predicted SNR meets sensitivity threshold (with 2σ margin)
5. **Fallback** to SF12/TX14 if action is unsafe

### 3. Downlink Execution

1. **Generates** `LinkADRReq` MAC command bytes: `[0x03, DR_TX, ChMask, Redundancy]`
2. **Publishes** to MQTT topic `application/{app_id}/device/{dev_eui}/command/down` on FPort 0
3. **ChirpStack** encrypts the MAC command and schedules downlink
4. **Node** receives and applies the new DR/TX settings automatically

### 4. Data Logging

Every transition is logged to `data/phase1_transitions.csv`:

```csv
rssi,snr,temp,hum,pres,dr,tx,reward
-110,5.5,25.34,60.5,1013.25,3,10,0.625
```

**Reward Calculation:**
```
reward = 0.5 * (dr / 5) + 0.5 * ((14 - tx) / 12)
```
- Higher DR (SF7 = DR5) → Higher reward
- Lower TX (2 dBm) → Higher reward

## Verification

### Check MQTT Messages

```bash
# Subscribe to all application topics
docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/#" -v
```

### Check CSV File

```bash
# View CSV contents
docker exec -it chirpstack-docker-adr-engine-1 cat /app/data/phase1_transitions.csv

# Or from host (Windows)
type data\phase1_transitions.csv
```

### Check Downlinks

```bash
# Subscribe to downlink commands
docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/+/device/+/command/down" -v
```

## Troubleshooting

### ADR Engine Not Starting

```bash
# Check logs for errors
docker-compose logs adr-engine

# Common issues:
# - Model files not found: Ensure Model/ directory is present
# - MQTT connection failed: Check mosquitto service is running
```

### No Uplinks Received

```bash
# Verify MQTT is receiving messages
docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/+/device/+/event/up" -v

# If no messages:
# - Check device is joined and sending data
# - Verify gateway is connected to ChirpStack
# - Check ChirpStack logs: docker-compose logs chirpstack
```

### CSV Not Being Created

```bash
# Check data directory permissions
ls -la data/

# Check ADR engine logs for CSV errors
docker-compose logs adr-engine | grep CSV
```

## Configuration

Environment variables in `docker-compose.yml`:

- `MQTT_BROKER_HOST`: MQTT broker hostname (default: `mosquitto`)
- `MQTT_BROKER_PORT`: MQTT broker port (default: `1883`)
- `APPLICATION_ID`: ChirpStack application ID (default: `1`)
- `CSV_OUTPUT_PATH`: CSV file path (default: `/app/data/phase1_transitions.csv`)

## Safety Shield Parameters

Defined in `adr_engine_phase1.py`:

- **Sensitivity Thresholds** (EU868):
  - SF7: -123 dBm
  - SF8: -126 dBm
  - SF9: -129 dBm
  - SF10: -132 dBm
  - SF11: -134.5 dBm
  - SF12: -137 dBm

- **Safety Margin**: 2σ (2 standard deviations)
- **Additional Margin**: 5 dB on top of sensitivity

## Next Steps

After collecting data for 48+ hours:

1. **Analyze CSV**: Use the logged transitions for Reinforcement Learning
2. **Train RL Model**: Use the reward signal to train a policy
3. **Phase 2**: Replace exploration with trained RL agent

## Support

For issues or questions:
- Check logs: `docker-compose logs -f adr-engine`
- Verify MQTT connectivity
- Ensure Device Profile has ADR disabled
- Confirm JavaScript decoder is configured correctly
