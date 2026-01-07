// ChirpStack JavaScript Codec for BME280 Sensor
// Decodes 6-byte payload: [TempH, TempL, HumH, HumL, PresH, PresL]
// Values are scaled by 100 (e.g., 2534 = 25.34°C)

function decodeUplink(input) {
  var bytes = input.bytes;
  
  // Validate payload length
  if (bytes.length !== 6) {
    return {
      errors: ["Invalid payload length. Expected 6 bytes, got " + bytes.length]
    };
  }
  
  // Decode temperature (bytes 0-1)
  var tempRaw = (bytes[0] << 8) | bytes[1];
  var temp = tempRaw / 100.0;
  
  // Decode humidity (bytes 2-3)
  var humRaw = (bytes[2] << 8) | bytes[3];
  var hum = humRaw / 100.0;
  
  // Decode pressure (bytes 4-5)
  var presRaw = (bytes[4] << 8) | bytes[5];
  var pres = presRaw / 100.0;
  
  return {
    data: {
      temp: temp,
      hum: hum,
      pres: pres
    }
  };
}

// Test function (optional, for debugging)
function test() {
  // Example: temp=25.34°C, hum=60.50%, pres=1013.25 hPa
  // Encoded: [0x09, 0xE6, 0x17, 0x9A, 0x27, 0x65]
  var testBytes = [0x09, 0xE6, 0x17, 0x9A, 0x27, 0x65];
  var result = decodeUplink({bytes: testBytes});
  console.log(JSON.stringify(result));
  // Expected: {"data":{"temp":25.34,"hum":60.5,"pres":1013.25}}
}
