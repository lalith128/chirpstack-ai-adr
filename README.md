# ChirpStack AI-ADR: AI-Driven Adaptive Data Rate for LoRaWAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ChirpStack v4](https://img.shields.io/badge/ChirpStack-v4-blue.svg)](https://www.chirpstack.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An intelligent, external Adaptive Data Rate (ADR) engine for ChirpStack v4 that uses AI-based "Safety Shield" to optimize LoRaWAN network performance while collecting data for Reinforcement Learning.

## 🎯 Overview

This project implements a **Phase 1 Data Collection System** for AI-driven ADR optimization in LoRaWAN networks. Instead of using ChirpStack's built-in ADR algorithm, this external engine:

- 🤖 Uses a **TensorFlow Lite model** to predict SNR for different DR/TX settings
- 🛡️ Implements a **Safety Shield** to prevent device disconnection
- 🔄 Explores different configurations using weighted random selection
- 📊 Logs all transitions to CSV for future Reinforcement Learning training
- 🚀 Sends MAC commands via MQTT for seamless integration

## 🏗️ Architecture

```
┌─────────────────┐
│  End Node       │
│  (Arduino LMIC) │
└────────┬────────┘
         │ LoRa Uplink
         ▼
┌─────────────────┐
│  LoRa Gateway   │
└────────┬────────┘
         │ UDP
         ▼
┌─────────────────────────────────────────────────────┐
│  ChirpStack v4 (Docker)                             │
│  ┌──────────────┐    ┌─────────────┐               │
│  │ JS Decoder   │───▶│ MQTT Broker │               │
│  │ (BME280)     │    │ (mosquitto) │               │
│  └──────────────┘    └──────┬──────┘               │
│                             │                       │
└─────────────────────────────┼───────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  ADR Engine      │
                    │  (Python)        │
                    │                  │
                    │  ┌────────────┐  │
                    │  │ TFLite     │  │
                    │  │ Model      │  │
                    │  └────────────┘  │
                    │                  │
                    │  ┌────────────┐  │
                    │  │ Safety     │  │
                    │  │ Shield     │  │
                    │  └────────────┘  │
                    │                  │
                    │  ┌────────────┐  │
                    │  │ CSV Logger │  │
                    │  └────────────┘  │
                    └──────────────────┘
```

## ✨ Features

### Phase 1: Data Collection
- ✅ **AI Safety Shield**: Predicts SNR using TensorFlow Lite model before applying DR/TX changes
- ✅ **Exploration Strategy**: Weighted random selection favoring efficient settings (SF7-SF12, TX 2-14 dBm)
- ✅ **MAC Command Generation**: Constructs standard LoRaWAN `LinkADRReq` commands
- ✅ **CSV Data Logging**: Records RSSI, SNR, sensor data, DR, TX, and reward scores
- ✅ **Robust Error Handling**: Graceful handling of JSON parsing, MQTT disconnections, and missing data
- ✅ **Docker Deployment**: Fully containerized with docker-compose

### Safety Shield
The Safety Shield ensures network reliability by:
1. Loading a pre-trained TFLite model and Scikit-Learn scalers
2. Predicting SNR for proposed DR/TX settings
3. Validating: `Predicted_SNR - 2σ >= Sensitivity_Threshold + 5dB margin`
4. Falling back to SF12/TX14 if action is unsafe

### Reward Function
```python
reward = 0.5 * (dr / 5) + 0.5 * ((14 - tx) / 12)
```
- Higher DR (SF7 = DR5) → Higher reward (faster transmission)
- Lower TX power → Higher reward (energy efficiency)

## 📋 Prerequisites

- **Hardware**:
  - Raspberry Pi 5 (or similar Linux system)
  - LoRa Gateway
  - Arduino-based end node with BME280 sensor
  - LoRa module (e.g., RFM95W)

- **Software**:
  - Docker & Docker Compose
  - ChirpStack v4 (included in docker-compose)
  - Python 3.10+ (in Docker container)
  - WSL2 (if running on Windows)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/lalith128/chirpstack-ai-adr.git
cd chirpstack-ai-adr
```

### 2. Start the Stack

```bash
docker-compose up -d --build
```

This will start:
- ChirpStack Server (port 8080)
- MQTT Broker (port 1883)
- PostgreSQL Database
- Redis Cache
- **ADR Engine** (Python)

### 3. Configure ChirpStack

1. **Open ChirpStack UI**: http://localhost:8080
   - Default credentials: `admin` / `admin`

2. **Create Application & Device Profile**

3. **Add JavaScript Decoder**:
   - Navigate to Device Profile → Codec
   - Set **Payload codec** to "JavaScript"
   - Copy contents from `codec_bme280.js`

4. **Native ADR is Already Disabled** ✅
   - All region files have `adr_disabled=true`

### 4. Upload Arduino Sketch

1. Open `examples/arduino_bme280_node/arduino_bme280_node.ino`
2. Update your device credentials:
   ```cpp
   static const u1_t PROGMEM APPEUI[8] = { ... };
   static const u1_t PROGMEM DEVEUI[8] = { ... };
   static const u1_t PROGMEM APPKEY[16] = { ... };
   ```
3. Install required libraries:
   - MCCI LoRaWAN LMIC library
   - Adafruit BME280
4. Upload to your Arduino board

### 5. Monitor the System

```bash
# View ADR engine logs
docker-compose logs -f adr-engine

# View all MQTT messages
docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/#" -v

# Check CSV data
cat data/phase1_transitions.csv
```

## 📁 Project Structure

```
chirpstack-ai-adr/
├── adr_engine/
│   ├── adr_engine_phase1.py      # Main ADR engine
│   ├── requirements.txt           # Python dependencies
│   └── Dockerfile                 # Docker image for ADR engine
├── models/
│   ├── lora_env_shield_float32.tflite  # TFLite model
│   ├── scaler_p0_context.pkl           # Context scaler
│   ├── scaler_p0_env.pkl               # Environment scaler
│   ├── scaler_p0_target.pkl            # Target scaler
│   └── encoder_p0_action.pkl           # Action encoder
├── codec/
│   └── codec_bme280.js            # ChirpStack JavaScript decoder
├── examples/
│   └── arduino_bme280_node/
│       └── arduino_bme280_node.ino     # Arduino example
├── docs/
│   ├── QUICKSTART.md              # Quick start guide
│   ├── ARCHITECTURE.md            # Detailed architecture
│   └── TROUBLESHOOTING.md         # Common issues & solutions
├── data/                          # CSV output directory
│   └── phase1_transitions.csv    # Logged transitions
├── docker-compose.yml             # Docker Compose configuration
└── README.md                      # This file
```

## 📊 Data Collection

The ADR engine logs every transition to `data/phase1_transitions.csv`:

```csv
rssi,snr,temp,hum,pres,dr,tx,reward
-110,5.5,25.34,60.5,1013.25,3,10,0.625
-108,6.2,25.41,60.3,1013.18,5,8,0.875
```

**Columns**:
- `rssi`: Received Signal Strength Indicator (dBm)
- `snr`: Signal-to-Noise Ratio (dB)
- `temp`: Temperature (°C)
- `hum`: Humidity (%)
- `pres`: Pressure (hPa)
- `dr`: Data Rate (0-5, where 5=SF7, 0=SF12)
- `tx`: Transmit Power (dBm)
- `reward`: Calculated reward score (0.0-1.0)

## 🔧 Configuration

### Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - MQTT_BROKER_HOST=mosquitto      # MQTT broker hostname
  - MQTT_BROKER_PORT=1883            # MQTT broker port
  - APPLICATION_ID=1                 # ChirpStack application ID
  - CSV_OUTPUT_PATH=/app/data/phase1_transitions.csv
```

### Safety Shield Parameters

Edit `adr_engine/adr_engine_phase1.py`:

```python
# Sensitivity thresholds (dBm) for EU868
SENSITIVITY = {
    7: -123.0,   # SF7
    8: -126.0,   # SF8
    9: -129.0,   # SF9
    10: -132.0,  # SF10
    11: -134.5,  # SF11
    12: -137.0   # SF12
}

# Safety margin
SAFETY_MARGIN_SIGMA = 2.0  # 2 standard deviations
```

### Exploration Strategy

Modify DR weights in `adr_engine_phase1.py`:

```python
# Weight distribution: SF7=30%, SF8=25%, SF9=20%, SF10=15%, SF11=5%, SF12=5%
dr_weights = [0.05, 0.05, 0.15, 0.20, 0.25, 0.30]  # DR0-DR5
```

## 🧪 Testing

### Verify Services

```bash
docker-compose ps
```

All services should show "Up" status.

### Check ADR Engine Logs

```bash
docker-compose logs adr-engine
```

Expected output:
```
INFO - Loading TFLite model and scalers...
INFO - Model and scalers loaded successfully
INFO - Connected to MQTT broker
INFO - Subscribed to application/+/device/+/event/up
```

### Monitor Uplinks

```bash
docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/+/device/+/event/up" -v
```

### Monitor Downlinks

```bash
docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/+/device/+/command/down" -v
```

## 🐛 Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues and solutions.

**Quick fixes**:

- **No uplinks received**: Check device join status, gateway connection
- **ADR engine not responding**: Check logs with `docker-compose logs adr-engine`
- **CSV not updating**: Verify file permissions and check for errors in logs

## 📈 Roadmap

### Phase 1: Data Collection ✅ (Current)
- [x] External ADR engine with Safety Shield
- [x] Exploration-based DR/TX selection
- [x] CSV logging for RL training
- [x] Docker deployment

### Phase 2: Reinforcement Learning (Planned)
- [ ] Train DQN/PPO agent using collected data
- [ ] Replace exploration with trained policy
- [ ] Online learning and adaptation
- [ ] Multi-device support

### Phase 3: Production Deployment (Future)
- [ ] A/B testing framework
- [ ] Performance metrics dashboard
- [ ] Auto-scaling and load balancing
- [ ] Cloud deployment support

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ChirpStack](https://www.chirpstack.io/) - Open-source LoRaWAN Network Server
- [MCCI LoRaWAN LMIC](https://github.com/mcci-catena/arduino-lmic) - Arduino LoRaWAN library
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Machine learning framework

## 📧 Contact

**Lalith V**
- GitHub: [@lalith128](https://github.com/lalith128)
- Repository: [chirpstack-ai-adr](https://github.com/lalith128/chirpstack-ai-adr)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ for the LoRaWAN community**
