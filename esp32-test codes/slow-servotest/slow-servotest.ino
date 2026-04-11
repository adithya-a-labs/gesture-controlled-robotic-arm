#include <Arduino.h>

int servoPin = 18;
int pwmChannel = 0;

int currentAngle = 90;

// Convert angle → duty
uint32_t angleToDuty(int angle) {
  int minDuty = 1638;
  int maxDuty = 8192;
  return map(angle, 0, 180, minDuty, maxDuty);
}

void setup() {
  Serial.begin(9600);

  // NEW API (ESP32 v3+)
  ledcAttach(servoPin, 50, 16);   // pin, freq, resolution

  ledcWrite(servoPin, angleToDuty(currentAngle))

  Serial.println("Enter angle (0-180):");
}

void loop() {
  if (Serial.available()) {
    int target = Serial.parseInt();

    if (target >= 0 && target <= 180) {
      moveSmooth(target);
    }

    while (Serial.available()) Serial.read();
  }
}

void moveSmooth(int target) {
  if (target > currentAngle) {
    for (int pos = currentAngle; pos <= target; pos++) {
      ledcWrite(servoPin, angleToDuty(pos));
      delay(15);
    }
  } else {
    for (int pos = currentAngle; pos >= target; pos--) {
      ledcWrite(servoPin, angleToDuty(pos));
      delay(15);
    }
  }

  currentAngle = target;
}