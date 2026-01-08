#!/usr/bin/env python3
"""
Phase 1 AI-Driven LoRaWAN Data Collector with External ADR Engine
Uses AI "Safety Shield" to explore DR/TX settings and log results to CSV
"""

import os
import sys
import json
import time
import base64
import random
import logging
from pathlib import Path
from datetime import datetime

import paho.mqtt.client as mqtt
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
APPLICATION_ID = os.getenv("APPLICATION_ID", "1")
CSV_OUTPUT_PATH = os.getenv("CSV_OUTPUT_PATH", "/app/data/phase1_transitions.csv")

# Model paths
MODEL_DIR = Path("/app/models")
TFLITE_MODEL_PATH = MODEL_DIR / "lora_env_shield_float32.tflite"
SCALER_CONTEXT_PATH = MODEL_DIR / "scaler_p0_context.pkl"
SCALER_ENV_PATH = MODEL_DIR / "scaler_p0_env.pkl"
SCALER_TARGET_PATH = MODEL_DIR / "scaler_p0_target.pkl"
ENCODER_ACTION_PATH = MODEL_DIR / "encoder_p0_action.pkl"

# LoRaWAN Parameters (EU868)
DR_TO_SF = {0: 12, 1: 11, 2: 10, 3: 9, 4: 8, 5: 7}
SF_TO_DR = {12: 0, 11: 1, 10: 2, 9: 3, 8: 4, 7: 5}

# Sensitivity thresholds (dBm) for EU868
SENSITIVITY = {
    7: -123.0,
    8: -126.0,
    9: -129.0,
    10: -132.0,
    11: -134.5,
    12: -137.0
}

# TX Power range (dBm)
TX_POWER_MIN = 2
TX_POWER_MAX = 14

# Safety margin (dB)
SAFETY_MARGIN_SIGMA = 2.0


class SafetyShield:
    """AI Safety Shield for predicting SNR and validating actions"""
    
    def __init__(self):
        logger.info("Loading TFLite model and scalers...")
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load scalers
        self.scaler_context = joblib.load(SCALER_CONTEXT_PATH)
        self.scaler_env = joblib.load(SCALER_ENV_PATH)
        self.scaler_target = joblib.load(SCALER_TARGET_PATH)
        
        # Load action encoder (if available)
        if ENCODER_ACTION_PATH.exists():
            self.encoder_action = joblib.load(ENCODER_ACTION_PATH)
        else:
            self.encoder_action = None
            logger.warning("Action encoder not found, using manual encoding")
        
        logger.info("Model and scalers loaded successfully")
    
    def predict_snr(self, rssi, snr, temp, hum, pres, dr, tx):
        """
        Predict SNR for given context and action
        
        Returns:
            tuple: (predicted_snr, uncertainty_sigma)
        """
        try:
            # Prepare context features (current state)
            # Assuming context = [rssi, snr, temp, hum, pres]
            context = np.array([[rssi, snr, temp, hum, pres]], dtype=np.float32)
            context_scaled = self.scaler_context.transform(context)
            
            # Prepare action features (DR, TX)
            # Assuming action = [dr, tx]
            action = np.array([[dr, tx]], dtype=np.float32)
            
            # Concatenate context and action
            # Model input shape: [batch, context_dim + action_dim]
            model_input = np.concatenate([context_scaled, action], axis=1).astype(np.float32)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], model_input)
            self.interpreter.invoke()
            
            # Get output (predicted SNR)
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_snr_scaled = output[0][0]
            
            # Inverse transform to get actual SNR
            predicted_snr = self.scaler_target.inverse_transform([[predicted_snr_scaled]])[0][0]
            
            # Estimate uncertainty (simplified: use 2 dB as default sigma)
            # In a real system, you'd have a separate uncertainty model
            uncertainty_sigma = 2.0
            
            return predicted_snr, uncertainty_sigma
            
        except Exception as e:
            logger.error(f"Error predicting SNR: {e}")
            return None, None
    
    def is_safe_action(self, rssi, snr, temp, hum, pres, dr, tx):
        """
        Check if action is safe using Safety Shield
        
        Returns:
            tuple: (is_safe, predicted_snr)
        """
        predicted_snr, sigma = self.predict_snr(rssi, snr, temp, hum, pres, dr, tx)
        
        if predicted_snr is None:
            logger.warning("Failed to predict SNR, rejecting action")
            return False, None
        
        # Get sensitivity threshold for the DR
        sf = DR_TO_SF[dr]
        sensitivity_threshold = SENSITIVITY[sf]
        
        # Safety rule: Predicted SNR - 2*Sigma >= Sensitivity Threshold
        # Note: SNR is relative to noise floor, sensitivity is absolute power
        # We need to check if RSSI + Predicted_SNR >= Sensitivity_Threshold
        predicted_rssi = rssi  # Simplified: assume RSSI stays similar
        
        # Conservative check: use lower bound of prediction
        lower_bound_snr = predicted_snr - SAFETY_MARGIN_SIGMA * sigma
        
        # Check if signal will be decodable
        # For safety, we check if current RSSI is above sensitivity + margin
        is_safe = (rssi + lower_bound_snr) >= (sensitivity_threshold + 5)  # 5 dB margin
        
        logger.info(f"Safety check: DR={dr} (SF{sf}), TX={tx}, "
                   f"Predicted SNR={predicted_snr:.2f}±{sigma:.2f}, "
                   f"Sensitivity={sensitivity_threshold}, Safe={is_safe}")
        
        return is_safe, predicted_snr


class ADREngine:
    """External ADR Engine with Exploration Strategy"""
    
    def __init__(self):
        self.safety_shield = SafetyShield()
        self.csv_initialized = False
        self.mqtt_client = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(CSV_OUTPUT_PATH), exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(CSV_OUTPUT_PATH):
            self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with header"""
        df = pd.DataFrame(columns=['rssi', 'snr', 'temp', 'hum', 'pres', 'current_dr', 'current_tx', 'target_dr', 'target_tx', 'action_type', 'reward'])
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        logger.info(f"Initialized CSV file: {CSV_OUTPUT_PATH}")
        self.csv_initialized = True
    
    def explore_action(self):
        """
        Exploration strategy: randomly pick efficient settings
        
        Returns:
            tuple: (dr, tx)
        """
        # Favor higher DR (lower SF) for exploration
        # Weight distribution: SF7=30%, SF8=25%, SF9=20%, SF10=15%, SF11=5%, SF12=5%
        dr_weights = [0.05, 0.05, 0.15, 0.20, 0.25, 0.30]  # DR0-DR5
        dr = random.choices(range(6), weights=dr_weights)[0]
        
        # Favor lower TX power for energy efficiency
        # Range: 2-14 dBm, favor lower values
        tx = random.randint(TX_POWER_MIN, TX_POWER_MAX)
        
        return dr, tx
    
    def fallback_safe_action(self):
        """
        Fallback to safest action: SF12, TX14
        
        Returns:
            tuple: (dr, tx)
        """
        return 0, 14  # DR0 = SF12, TX = 14 dBm
    
    def calculate_reward(self, dr, tx):
        """
        Calculate reward score (0.0 to 1.0)
        High DR = High Reward, Low TX = High Reward
        
        Returns:
            float: reward score
        """
        dr_reward = dr / 5.0  # DR5 (SF7) = 1.0, DR0 (SF12) = 0.0
        tx_reward = (TX_POWER_MAX - tx) / (TX_POWER_MAX - TX_POWER_MIN)  # TX2 = 1.0, TX14 = 0.0
        
        # Weighted combination
        reward = 0.5 * dr_reward + 0.5 * tx_reward
        return reward
    
    def generate_linkadr_req(self, dr, tx):
        """
        Generate LinkADRReq MAC command bytes
        
        Format: [CID, DR_TX, ChMask_LSB, ChMask_MSB, Redundancy]
        
        Returns:
            bytes: MAC command payload
        """
        cid = 0x03  # LinkADRReq
        
        # DR_TX byte: [7:4] = TXPower, [3:0] = DataRate
        # TXPower encoding: 0 = MaxEIRP, 1 = MaxEIRP-2, ..., 7 = MaxEIRP-14
        # For EU868: MaxEIRP = 16 dBm (14 dBm conducted + 2.15 dBi antenna)
        # Simplified: use direct mapping
        tx_power_index = max(0, min(7, (14 - tx) // 2))
        dr_tx_byte = (tx_power_index << 4) | (dr & 0x0F)
        
        # ChMask: Enable all channels (0xFFFF for EU868 default channels)
        ch_mask_lsb = 0xFF
        ch_mask_msb = 0xFF
        
        # Redundancy: [7] = RFU, [6:4] = ChMaskCntl, [3:0] = NbTrans
        # ChMaskCntl = 0 (channels 0-15), NbTrans = 1 (no retransmissions)
        redundancy = 0x01
        
        payload = bytes([cid, dr_tx_byte, ch_mask_lsb, ch_mask_msb, redundancy])
        
        logger.info(f"Generated LinkADRReq: DR={dr}, TX={tx}dBm, Payload={payload.hex()}")
        return payload
    
    def publish_downlink(self, dev_eui, payload):
        """
        Publish downlink MAC command to MQTT
        
        Args:
            dev_eui: Device EUI
            payload: MAC command bytes
        """
        topic = f"application/{APPLICATION_ID}/device/{dev_eui}/command/down"
        
        # ChirpStack downlink format
        downlink_msg = {
            "devEui": dev_eui,
            "confirmed": False,
            "fPort": 0,  # FPort 0 = MAC commands
            "data": base64.b64encode(payload).decode('utf-8')
        }
        
        try:
            self.mqtt_client.publish(topic, json.dumps(downlink_msg), qos=1)
            logger.info(f"Published downlink to {dev_eui}: {downlink_msg}")
        except Exception as e:
            logger.error(f"Failed to publish downlink: {e}")
    
    def log_to_csv(self, rssi, snr, temp, hum, pres, current_dr, current_tx, target_dr, target_tx, action_type, reward):
        """
        Log transition to CSV file
        
        Args:
            rssi: Received signal strength
            snr: Signal-to-noise ratio
            temp: Temperature from sensor
            hum: Humidity from sensor
            pres: Pressure from sensor
            current_dr: Actual DR used by device (from txInfo)
            current_tx: Estimated current TX power
            target_dr: Target DR selected by exploration
            target_tx: Target TX power selected by exploration
            action_type: 'active_change' or 'passive_maintain'
            reward: Calculated reward score
        """
        try:
            row = {
                'rssi': rssi,
                'snr': snr,
                'temp': temp,
                'hum': hum,
                'pres': pres,
                'current_dr': current_dr,
                'current_tx': current_tx,
                'target_dr': target_dr,
                'target_tx': target_tx,
                'action_type': action_type,
                'reward': reward
            }
            
            df = pd.DataFrame([row])
            df.to_csv(CSV_OUTPUT_PATH, mode='a', header=False, index=False)
            
            logger.info(f"Logged to CSV [{action_type}]: current_dr={current_dr}, target_dr={target_dr}, reward={reward:.3f}")
        except Exception as e:
            logger.error(f"Failed to log to CSV: {e}")

    
    def process_uplink(self, msg_payload):
        """
        Process uplink message and execute ADR logic with hardened safeguards.
        
        Safeguards:
        1. Stale State Handling: Extract actual DR from txInfo (not local state)
        2. Duty Cycle Optimization: Only send downlink if DR/TX differs from current
        """
        try:
            data = json.loads(msg_payload)
            
            # Extract device EUI
            dev_eui = data.get('devEui', data.get('deviceInfo', {}).get('devEui'))
            if not dev_eui:
                logger.warning("No devEui found in message")
                return
            
            # Extract RX info
            rx_info = data.get('rxInfo', [])
            if not rx_info:
                logger.warning("No rxInfo found in message")
                return
            
            rssi = rx_info[0].get('rssi', 0)
            snr = rx_info[0].get('snr', 0)
            
            # ================================================================
            # SAFEGUARD 1: Handle Stale State (Device Reboots)
            # Extract actual DR from txInfo - DO NOT rely on local state
            # ================================================================
            tx_info = data.get('txInfo', {})
            
            # ChirpStack v4 JSON path: txInfo.modulation.lora.spreadingFactor
            # or txInfo.dr (depending on version)
            current_dr = None
            
            # Try to get DR directly
            if 'dr' in tx_info:
                current_dr = tx_info.get('dr')
            elif 'dataRate' in tx_info:
                current_dr = tx_info.get('dataRate')
            
            # Try to extract from modulation info (ChirpStack v4 format)
            if current_dr is None:
                modulation = tx_info.get('modulation', {})
                lora_modulation = modulation.get('lora', {})
                sf = lora_modulation.get('spreadingFactor')
                if sf and sf in SF_TO_DR:
                    current_dr = SF_TO_DR[sf]
            
            # Fallback: try to parse from the data rate index in other fields
            if current_dr is None:
                # Check in the root of the message
                current_dr = data.get('dr', data.get('dataRate'))
            
            if current_dr is None:
                logger.warning("Could not extract current DR from uplink, using default DR0 (SF12)")
                current_dr = 0  # Safe fallback to SF12
            
            # Estimate current TX power (not directly available in uplink)
            # Use a conservative estimate based on RSSI and typical path loss
            # For now, assume TX_POWER_MAX as we can't know the actual TX power
            current_tx = TX_POWER_MAX  # Conservative estimate
            
            logger.info(f"Extracted from txInfo: current_dr={current_dr} (SF{DR_TO_SF.get(current_dr, '?')})")
            
            # Extract decoded object
            decoded_object = data.get('object', {})
            if not decoded_object:
                logger.warning("No decoded object found in message")
                return
            
            temp = decoded_object.get('temp', 0)
            hum = decoded_object.get('hum', 0)
            pres = decoded_object.get('pres', 0)
            
            logger.info(f"Received uplink from {dev_eui}: RSSI={rssi}, SNR={snr}, "
                       f"DR={current_dr} (SF{DR_TO_SF.get(current_dr, '?')}), "
                       f"Temp={temp}, Hum={hum}, Pres={pres}")
            
            # Exploration: pick a random target action
            target_dr, target_tx = self.explore_action()
            
            # Safety Shield: validate target action
            is_safe, predicted_snr = self.safety_shield.is_safe_action(
                rssi, snr, temp, hum, pres, target_dr, target_tx
            )
            
            # If not safe, fallback to safest action
            if not is_safe:
                logger.warning(f"Action DR={target_dr}, TX={target_tx} is not safe, falling back")
                target_dr, target_tx = self.fallback_safe_action()
            
            # Calculate reward based on TARGET settings (what we want to achieve)
            reward = self.calculate_reward(target_dr, target_tx)
            
            # ================================================================
            # SAFEGUARD 2: Duty Cycle Optimization (Smart Exploration)
            # Only send downlink if target differs from current
            # ================================================================
            dr_changed = (target_dr != current_dr)
            tx_changed = (target_tx != current_tx)
            settings_differ = dr_changed or tx_changed
            
            if settings_differ:
                # ACTIVE CHANGE: Target differs from current - send downlink
                action_type = "active_change"
                
                # Generate MAC command
                mac_payload = self.generate_linkadr_req(target_dr, target_tx)
                
                # Publish downlink
                self.publish_downlink(dev_eui, mac_payload)
                
                logger.info(f"Active Change: DR {current_dr}->{target_dr}, TX {current_tx}->{target_tx}")
            else:
                # PASSIVE MAINTAIN: Current settings are optimal - skip downlink
                action_type = "passive_maintain"
                
                logger.info(f"Passive Maintain: Current DR={current_dr}, TX={current_tx} matches target. "
                           f"Skipping downlink to save airtime.")
            
            # ALWAYS log to CSV - both active changes and passive maintains are valid training data
            self.log_to_csv(
                rssi=rssi,
                snr=snr,
                temp=temp,
                hum=hum,
                pres=pres,
                current_dr=current_dr,
                current_tx=current_tx,
                target_dr=target_dr,
                target_tx=target_tx,
                action_type=action_type,
                reward=reward
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except KeyError as e:
            logger.error(f"Missing key in message: {e}")
        except Exception as e:
            logger.error(f"Error processing uplink: {e}", exc_info=True)



def on_connect(client, userdata, flags, rc):
    """MQTT on_connect callback"""
    if rc == 0:
        logger.info("Connected to MQTT broker")
        
        # Subscribe to uplink events
        topic = f"application/+/device/+/event/up"
        client.subscribe(topic)
        logger.info(f"Subscribed to {topic}")
    else:
        logger.error(f"Failed to connect to MQTT broker, return code {rc}")


def on_message(client, userdata, msg):
    """MQTT on_message callback"""
    logger.debug(f"Received message on topic {msg.topic}")
    
    # Get ADR engine from userdata
    adr_engine = userdata
    
    # Process uplink
    adr_engine.process_uplink(msg.payload)


def on_disconnect(client, userdata, rc):
    """MQTT on_disconnect callback"""
    if rc != 0:
        logger.warning(f"Unexpected disconnect from MQTT broker, return code {rc}")
        logger.info("Attempting to reconnect...")


def main():
    """Main entry point"""
    logger.info("Starting Phase 1 ADR Engine...")
    
    # Initialize ADR engine
    adr_engine = ADREngine()
    
    # Setup MQTT client
    mqtt_client = mqtt.Client(userdata=adr_engine)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.on_disconnect = on_disconnect
    
    # Store client reference in ADR engine
    adr_engine.mqtt_client = mqtt_client
    
    # Connect to MQTT broker
    logger.info(f"Connecting to MQTT broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
    
    try:
        mqtt_client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, 60)
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker: {e}")
        sys.exit(1)
    
    # Start MQTT loop
    logger.info("ADR Engine running. Press Ctrl+C to stop.")
    
    try:
        mqtt_client.loop_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down ADR Engine...")
        mqtt_client.disconnect()
        logger.info("ADR Engine stopped")


if __name__ == "__main__":
    main()
