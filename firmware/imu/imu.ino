#include <Wire.h>

const int MPU_addr = 0x68;
const int MAG_addr = 0x0C;

int16_t AcX, AcY, AcZ;
int16_t GyX, GyY, GyZ;
int16_t MagX, MagY, MagZ;

float ax, ay, az;
float gx, gy, gz;

unsigned long previousMicros = 0;
const long sampleInterval = 5000; // 200Hz

void setup() {
  Serial.begin(115200);

  Wire.begin(0, 2);

  // Wake MPU9250
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);
  Wire.write(0x00);
  Wire.endTransmission(true);

  // Accelerometer ±16g
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1C);
  Wire.write(0x18);
  Wire.endTransmission(true);

  // Gyroscope ±2000 dps  ← CHANGED from ±250 dps
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1B);
  Wire.write(0x18);  // ← CHANGED from 0x00 to 0x18 (±2000 dps)
  Wire.endTransmission(true);

  // Enable bypass for magnetometer
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x37);
  Wire.write(0x02);
  Wire.endTransmission(true);

  // AK8963 magnetometer 16-bit continuous mode
  Wire.beginTransmission(MAG_addr);
  Wire.write(0x0A);
  Wire.write(0x16);
  Wire.endTransmission(true);
}

void readMPU() {
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x3B);
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)MPU_addr, (size_t)14, true);

  if (Wire.available() >= 14) {
    AcX = Wire.read() << 8 | Wire.read();
    AcY = Wire.read() << 8 | Wire.read();
    AcZ = Wire.read() << 8 | Wire.read();

    Wire.read();
    Wire.read();

    GyX = Wire.read() << 8 | Wire.read();
    GyY = Wire.read() << 8 | Wire.read();
    GyZ = Wire.read() << 8 | Wire.read();
  }
}

void readMag() {
  Wire.beginTransmission(MAG_addr);
  Wire.write(0x03);
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)MAG_addr, (size_t)7, true);

  if (Wire.available() >= 7) {
    uint8_t xl = Wire.read();
    uint8_t xh = Wire.read();
    uint8_t yl = Wire.read();
    uint8_t yh = Wire.read();
    uint8_t zl = Wire.read();
    uint8_t zh = Wire.read();
    Wire.read();

    MagX = (int16_t)(xh << 8 | xl);
    MagY = (int16_t)(yh << 8 | yl);
    MagZ = (int16_t)(zh << 8 | zl);
  }
}

void loop() {
  unsigned long currentMicros = micros();

  if (currentMicros - previousMicros >= sampleInterval) {
    previousMicros = currentMicros;

    readMPU();
    readMag();

    // Accelerometer ±16g: 2048 LSB/g
    ax = AcX / 2048.0;
    ay = AcY / 2048.0;
    az = AcZ / 2048.0;

    // Gyroscope ±2000 dps: 16.4 LSB/(°/s)  ← CHANGED from 131.0
    gx = GyX / 16.4;
    gy = GyY / 16.4;
    gz = GyZ / 16.4;

    Serial.print(ax, 2);
    Serial.print(",");
    Serial.print(ay, 2);
    Serial.print(",");
    Serial.print(az, 2);
    Serial.print(",");
    Serial.print(gx, 2);
    Serial.print(",");
    Serial.print(gy, 2);
    Serial.print(",");
    Serial.print(gz, 2);
    Serial.print(",");
    Serial.print(MagX);
    Serial.print(",");
    Serial.print(MagY);
    Serial.print(",");
    Serial.println(MagZ);
  }
}