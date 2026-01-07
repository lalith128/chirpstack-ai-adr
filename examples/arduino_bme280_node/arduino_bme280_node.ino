/*
 * Arduino LoRaWAN Node - BME280 Sensor Example
 * 
 * This example shows how to send BME280 sensor data in the format
 * expected by the ChirpStack JavaScript decoder.
 * 
 * Payload Format: 6 bytes [TempH, TempL, HumH, HumL, PresH, PresL]
 * Values are scaled by 100 (e.g., 25.34°C = 2534)
 * 
 * Compatible with LMIC library and Phase 1 ADR Engine
 */

#include <lmic.h>
#include <hal/hal.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_BME280.h>

// LoRaWAN Configuration
// Replace with your device credentials from ChirpStack
static const u1_t PROGMEM APPEUI[8] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const u1_t PROGMEM DEVEUI[8] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };
static const u1_t PROGMEM APPKEY[16] = { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
                                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

void os_getArtEui (u1_t* buf) { memcpy_P(buf, APPEUI, 8);}
void os_getDevEui (u1_t* buf) { memcpy_P(buf, DEVEUI, 8);}
void os_getDevKey (u1_t* buf) { memcpy_P(buf, APPKEY, 16);}

// Pin mapping for your board (adjust as needed)
const lmic_pinmap lmic_pins = {
    .nss = 10,
    .rxtx = LMIC_UNUSED_PIN,
    .rst = 9,
    .dio = {2, 6, 7},
};

// BME280 sensor
Adafruit_BME280 bme;

// Transmission interval (seconds)
const unsigned TX_INTERVAL = 60;

// Job scheduler
static osjob_t sendjob;

// Payload buffer
static uint8_t payload[6];

void onEvent (ev_t ev) {
    Serial.print(os_getTime());
    Serial.print(": ");
    switch(ev) {
        case EV_SCAN_TIMEOUT:
            Serial.println(F("EV_SCAN_TIMEOUT"));
            break;
        case EV_BEACON_FOUND:
            Serial.println(F("EV_BEACON_FOUND"));
            break;
        case EV_BEACON_MISSED:
            Serial.println(F("EV_BEACON_MISSED"));
            break;
        case EV_BEACON_TRACKED:
            Serial.println(F("EV_BEACON_TRACKED"));
            break;
        case EV_JOINING:
            Serial.println(F("EV_JOINING"));
            break;
        case EV_JOINED:
            Serial.println(F("EV_JOINED"));
            // Disable link check validation (not needed for ADR)
            LMIC_setLinkCheckMode(0);
            break;
        case EV_JOIN_FAILED:
            Serial.println(F("EV_JOIN_FAILED"));
            break;
        case EV_REJOIN_FAILED:
            Serial.println(F("EV_REJOIN_FAILED"));
            break;
        case EV_TXCOMPLETE:
            Serial.println(F("EV_TXCOMPLETE (includes waiting for RX windows)"));
            if (LMIC.txrxFlags & TXRX_ACK)
              Serial.println(F("Received ack"));
            if (LMIC.dataLen) {
              Serial.print(F("Received "));
              Serial.print(LMIC.dataLen);
              Serial.println(F(" bytes of payload"));
              
              // MAC commands are handled automatically by LMIC
              // The ADR settings will be applied automatically
            }
            // Schedule next transmission
            os_setTimedCallback(&sendjob, os_getTime()+sec2osticks(TX_INTERVAL), do_send);
            break;
        case EV_LOST_TSYNC:
            Serial.println(F("EV_LOST_TSYNC"));
            break;
        case EV_RESET:
            Serial.println(F("EV_RESET"));
            break;
        case EV_RXCOMPLETE:
            Serial.println(F("EV_RXCOMPLETE"));
            break;
        case EV_LINK_DEAD:
            Serial.println(F("EV_LINK_DEAD"));
            break;
        case EV_LINK_ALIVE:
            Serial.println(F("EV_LINK_ALIVE"));
            break;
        case EV_TXSTART:
            Serial.println(F("EV_TXSTART"));
            break;
        case EV_TXCANCELED:
            Serial.println(F("EV_TXCANCELED"));
            break;
        case EV_RXSTART:
            /* do not print anything -- it wrecks timing */
            break;
        case EV_JOIN_TXCOMPLETE:
            Serial.println(F("EV_JOIN_TXCOMPLETE: no JoinAccept"));
            break;
        default:
            Serial.print(F("Unknown event: "));
            Serial.println((unsigned) ev);
            break;
    }
}

void do_send(osjob_t* j){
    // Check if there is not a current TX/RX job running
    if (LMIC.opmode & OP_TXRXPEND) {
        Serial.println(F("OP_TXRXPEND, not sending"));
    } else {
        // Read BME280 sensor
        float temp = bme.readTemperature();
        float hum = bme.readHumidity();
        float pres = bme.readPressure() / 100.0F; // Convert Pa to hPa
        
        Serial.print(F("Temperature: "));
        Serial.print(temp);
        Serial.println(F(" °C"));
        
        Serial.print(F("Humidity: "));
        Serial.print(hum);
        Serial.println(F(" %"));
        
        Serial.print(F("Pressure: "));
        Serial.print(pres);
        Serial.println(F(" hPa"));
        
        // Encode payload: scale by 100 and convert to 2 bytes each
        uint16_t temp_scaled = (uint16_t)(temp * 100);
        uint16_t hum_scaled = (uint16_t)(hum * 100);
        uint16_t pres_scaled = (uint16_t)(pres * 100);
        
        payload[0] = (temp_scaled >> 8) & 0xFF;  // TempH
        payload[1] = temp_scaled & 0xFF;         // TempL
        payload[2] = (hum_scaled >> 8) & 0xFF;   // HumH
        payload[3] = hum_scaled & 0xFF;          // HumL
        payload[4] = (pres_scaled >> 8) & 0xFF;  // PresH
        payload[5] = pres_scaled & 0xFF;         // PresL
        
        Serial.print(F("Payload: "));
        for (int i = 0; i < 6; i++) {
            if (payload[i] < 0x10) Serial.print(F("0"));
            Serial.print(payload[i], HEX);
            Serial.print(F(" "));
        }
        Serial.println();
        
        // Prepare upstream data transmission at the next possible time.
        // FPort = 1 (application data)
        LMIC_setTxData2(1, payload, sizeof(payload), 0);
        Serial.println(F("Packet queued"));
    }
    // Next TX is scheduled after TX_COMPLETE event.
}

void setup() {
    Serial.begin(115200);
    Serial.println(F("Starting LoRaWAN BME280 Node"));
    
    // Initialize BME280
    if (!bme.begin(0x76)) {  // Try address 0x76, change to 0x77 if needed
        Serial.println(F("Could not find BME280 sensor!"));
        while (1);
    }
    
    Serial.println(F("BME280 sensor initialized"));
    
    // LMIC init
    os_init();
    
    // Reset the MAC state. Session and pending data transfers will be discarded.
    LMIC_reset();
    
    // Start job (sending automatically starts OTAA too)
    do_send(&sendjob);
}

void loop() {
    os_runloop_once();
}
