#include <Arduino.h>

// ----------- SERVO PINS -----------
int pin_s4 = 27; // Base
int pin_s3 = 14; // Shoulder
int pin_s2 = 12; // Elbow
int pin_s1 = 25; // Gripper

// ----------- PWM SETTINGS -----------
int freq = 50;
int resolution = 16;

// ----------- ANGLE STORAGE -----------
int s1 = 0;
int s2 = 90;
int s3 = 90;
int s4 = 90;

// ----------- CONVERT ANGLE → DUTY -----------
uint32_t angleToDuty(int angle) {
  int minDuty = 1638;
  int maxDuty = 8192;
  return map(angle, 0, 180, minDuty, maxDuty);
}

// ----------- SETUP -----------
void setup() {
  Serial.begin(115200);

  // Attach all servos
  ledcAttach(pin_s1, freq, resolution);
  ledcAttach(pin_s2, freq, resolution);
  ledcAttach(pin_s3, freq, resolution);
  ledcAttach(pin_s4, freq, resolution);

  Serial.println("ESP32 Ready (Direct Servo Mode)");
}

// ----------- LOOP -----------
void loop() {

  // Read incoming serial data
  if (Serial.available()) {

    String data = Serial.readStringUntil('\n');

    int parsed = sscanf(data.c_str(), "%d,%d,%d,%d", &s4, &s3, &s2, &s1);

    if (parsed == 4) {

      // Directly apply angles (NO MODIFICATION)
      ledcWrite(pin_s4, angleToDuty(s4)); // Base
      ledcWrite(pin_s3, angleToDuty(s3)); // Shoulder
      ledcWrite(pin_s2, angleToDuty(s2)); // Elbow
      ledcWrite(pin_s1, angleToDuty(s1)); // Gripper

      // Debug (optional)
      Serial.print("Applied: ");
      Serial.print(s4); Serial.print(",");
      Serial.print(s3); Serial.print(",");
      Serial.print(s2); Serial.print(",");
      Serial.println(s1);
    }
  }
}