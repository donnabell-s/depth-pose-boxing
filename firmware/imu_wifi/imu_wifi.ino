#include <Wire.h>
#include <ESP8266WiFi.h>
#include <WiFiUdp.h>

// const char* ssid = "TouchLab";
// const char* password = "touchmenot";

// IPAddress pcIP(192,168,0,148);


// const char* ssid = "OPPO";
// const char* password = "hotpot00";

// IPAddress pcIP(172, 20, 143, 146);

const char* ssid = "hotpot";
const char* password = "kastellan";

IPAddress pcIP(192, 168, 137, 79);

const int port = 4210;

WiFiUDP udp;

const int MPU_addr = 0x68;
const int MAG_addr = 0x0C;

int16_t AcX, AcY, AcZ;
int16_t GyX, GyY, GyZ;
int16_t MagX, MagY, MagZ;

float ax, ay, az;

unsigned long previousMicros = 0;
const long sampleInterval = 5000;  // 200Hz

// Batching parameters
const int BATCH_SIZE = 10;          // samples per packet
char batchBuffer[1400];             // UDP packet buffer
int batchPos = 0;                   // current write position in buffer
int batchCount = 0;                 // samples currently in buffer
unsigned long batchStartMicros = 0; // timestamp of first sample in current batch

void setup() {
  Serial.begin(115200);
  Wire.begin(0, 2);

  // Wake MPU
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  // Set ±16g
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x1C);
  Wire.write(0x18);
  Wire.endTransmission(true);

  // Enable bypass
  Wire.beginTransmission(MPU_addr);
  Wire.write(0x37);
  Wire.write(0x02);
  Wire.endTransmission(true);

  // Init AK8963 magnetometer
  Wire.beginTransmission(MAG_addr);
  Wire.write(0x0A);
  Wire.write(0x16);
  Wire.endTransmission(true);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
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
    Wire.read(); Wire.read();  // skip temp
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
    MagX = (xh << 8) | xl;
    MagY = (yh << 8) | yl;
    MagZ = (zh << 8) | zl;
  }
}

void flushBatch() {
  if (batchCount == 0) return;
  
  udp.beginPacket(pcIP, port);
  udp.write((const uint8_t*)batchBuffer, batchPos);
  udp.endPacket();
  
  batchPos = 0;
  batchCount = 0;
}

void loop() {
  unsigned long currentMicros = micros();

  if (currentMicros - previousMicros >= sampleInterval) {
    previousMicros = currentMicros;

    readMPU();
    readMag();

    // ±16g conversion (2048 LSB/g)
    ax = AcX / 2048.0;
    ay = AcY / 2048.0;
    az = AcZ / 2048.0;

    // Record start time of batch
    if (batchCount == 0) {
      batchStartMicros = currentMicros;
    }

    // Calculate relative offset within batch in microseconds
    unsigned long relMicros = currentMicros - batchStartMicros;

    // Append sample to batch buffer
    // Format per line: "relMicros,ax,ay,az,gx,gy,gz,mx,my,mz\n"
    int written = snprintf(
      batchBuffer + batchPos,
      sizeof(batchBuffer) - batchPos,
      "%lu,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d,%d\n",
      relMicros,
      ax, ay, az,
      GyX, GyY, GyZ,
      MagX, MagY, MagZ
    );

    if (written > 0) {
      batchPos += written;
      batchCount++;
    }

    // Send when batch is full
    if (batchCount >= BATCH_SIZE) {
      flushBatch();
    }
  }
}