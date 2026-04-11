#include <ESP32Servo.h>

Servo s1; // gripper

void setup() {
  Serial.begin(115200);

  s1.setPeriodHertz(50);
  s1.attach(25);

  delay(500);

  Serial.println("Enter angle (0–180):");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int angle = input.toInt();
    angle = constrain(angle, 0, 180);

    s1.write(angle);

    Serial.print("Moved to: ");
    Serial.println(angle);
  }
}