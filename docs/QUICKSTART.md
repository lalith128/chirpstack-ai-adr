# Phase 1 ADR Engine - Quick Start Guide

## ✅ System Status

**All services are running successfully!**

```
✅ ChirpStack Server (port 8080)
✅ MQTT Broker (port 1883)
✅ PostgreSQL Database
✅ Redis Cache
✅ ADR Engine (Python)
```

**ADR Engine Status:**
- ✅ TFLite model loaded
- ✅ Scalers loaded
- ✅ MQTT connected
- ✅ Subscribed to uplink events
- ✅ CSV file initialized

---

## Next Steps

### 1. Configure ChirpStack Device Profile

1. **Open ChirpStack UI**: http://localhost:8080
   - Default credentials: `admin` / `admin`

2. **Navigate to Device Profiles**
   - Applications → Your Application → Device Profiles

3. **Add JavaScript Decoder**
   - Edit your device profile
   - Go to "Codec" tab
   - Set **Payload codec** to "JavaScript"
   - Copy the contents of `codec_bme280.js` into the **Decoder** field
   - Click "Submit"

4. **Disable Native ADR** (CRITICAL)
   - In the same device profile, go to "ADR" section
   - Set **ADR algorithm** to "Disabled" OR uncheck "ADR enabled"
   - Click "Submit"

---

### 2. Upload Arduino Sketch

1. **Open** `arduino_bme280_example.ino` in Arduino IDE

2. **Update Credentials**:
   ```cpp
   static const u1_t PROGMEM APPEUI[8] = { ... };  // From ChirpStack
   static const u1_t PROGMEM DEVEUI[8] = { ... };  // From ChirpStack
   static const u1_t PROGMEM APPKEY[16] = { ... }; // From ChirpStack
   ```

3. **Install Libraries**:
   - LMIC (MCCI LoRaWAN LMIC library)
   - Adafruit BME280

4. **Upload** to your Arduino board

---

### 3. Monitor the System

#### Check ADR Engine Logs
```bash
wsl docker-compose logs -f adr-engine
```

**Expected output when device sends uplink:**
```
INFO - Received uplink from <dev_eui>: RSSI=-110, SNR=5.5, Temp=25.34, Hum=60.5, Pres=1013.25
INFO - Safety check: DR=3 (SF9), TX=10, Predicted SNR=6.2±2.0, Sensitivity=-129, Safe=True
INFO - Generated LinkADRReq: DR=3, TX=10dBm, Payload=03...
INFO - Published downlink to <dev_eui>
INFO - Logged to CSV: {'rssi': -110, 'snr': 5.5, ...}
```

#### Monitor MQTT Messages
```bash
wsl docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/#" -v
```

#### Check CSV Data
```bash
wsl docker exec chirpstack-docker-adr-engine-1 cat /app/data/phase1_transitions.csv
```

Or from Windows:
```powershell
type data\phase1_transitions.csv
```

---

### 4. Verify Downlinks

```bash
wsl docker exec -it chirpstack-docker-mosquitto-1 mosquitto_sub -t "application/+/device/+/command/down" -v
```

---

## Troubleshooting

### No Uplinks Received

1. **Check device is joined**:
   - Look for "EV_JOINED" in Arduino Serial Monitor
   - Check ChirpStack UI → Devices → Your Device → "LoRaWAN frames"

2. **Check gateway connection**:
   - ChirpStack UI → Gateways → Your Gateway → "Live LoRaWAN frames"

3. **Verify MQTT is working**:
   ```bash
   wsl docker-compose logs mosquitto
   ```

### ADR Engine Not Responding

1. **Check logs for errors**:
   ```bash
   wsl docker-compose logs adr-engine
   ```

2. **Restart the service**:
   ```bash
   wsl docker-compose restart adr-engine
   ```

3. **Verify MQTT connection**:
   ```bash
   wsl docker-compose logs adr-engine | grep "Connected to MQTT"
   ```

### CSV Not Updating

1. **Check file exists**:
   ```bash
   wsl docker exec chirpstack-docker-adr-engine-1 ls -la /app/data/
   ```

2. **Check permissions**:
   ```bash
   wsl docker exec chirpstack-docker-adr-engine-1 touch /app/data/test.txt
   ```

3. **Check logs for CSV errors**:
   ```bash
   wsl docker-compose logs adr-engine | grep CSV
   ```

---

## Useful Commands

### View All Services
```bash
wsl docker-compose ps
```

### Stop All Services
```bash
wsl docker-compose down
```

### Start All Services
```bash
wsl docker-compose up -d
```

### View Logs
```bash
# All services
wsl docker-compose logs -f

# Specific service
wsl docker-compose logs -f adr-engine
wsl docker-compose logs -f chirpstack
wsl docker-compose logs -f mosquitto
```

### Restart ADR Engine
```bash
wsl docker-compose restart adr-engine
```

### Access Container Shell
```bash
wsl docker exec -it chirpstack-docker-adr-engine-1 /bin/bash
```

---

## Data Collection

The system is now ready to collect data for 48+ hours. The CSV file will accumulate:
- RSSI and SNR measurements
- Sensor readings (temp, humidity, pressure)
- DR and TX power settings
- Reward scores for RL training

**Location**: `data/phase1_transitions.csv`

**Format**:
```csv
rssi,snr,temp,hum,pres,dr,tx,reward
-110,5.5,25.34,60.5,1013.25,3,10,0.625
```

---

## Support

For detailed documentation, see:
- **Setup Guide**: `README_ADR_PHASE1.md`
- **Arduino Example**: `arduino_bme280_example.ino`
- **JavaScript Decoder**: `codec_bme280.js`

---

## System is Ready! 🚀

The Phase 1 AI-Driven LoRaWAN Data Collector is fully operational and waiting for uplinks from your Arduino node.
