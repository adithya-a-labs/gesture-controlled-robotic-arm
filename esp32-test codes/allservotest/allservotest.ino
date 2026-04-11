#include <ESP32Servo.h>

// Servo objects
Servo s1; // gripper (GPIO 13)
Servo s2; // elbow   (GPIO 12)
Servo s3; // shoulder(GPIO 14)
Servo s4; // base    (GPIO 27)

// Pin definitions
const int pin_s1 = 25;
const int pin_s2 = 12;
const int pin_s3 = 14;
const int pin_s4 = 27;

void setup() {
  Serial.begin(115200);

  // Set servo PWM frequency (important for stability)
  s1.setPeriodHertz(50);
  s2.setPeriodHertz(50);
  s3.setPeriodHertz(50);
  s4.setPeriodHertz(50);

  // Attach servos
  s1.attach(pin_s1, 500, 2400);
  s2.attach(pin_s2, 500, 2400);
  s3.attach(pin_s3, 500, 2400);
  s4.attach(pin_s4, 500, 2400);

  Serial.println("Ready. Enter: s3,s4,s2,s1");
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int values[4];
    int index = 0;

    char buffer[50];
    input.toCharArray(buffer, 50);

    char *token = strtok(buffer, ",");

    while (token != NULL && index < 4) {
      values[index++] = atoi(token);
      token = strtok(NULL, ",");
    }

    if (index == 4) {
      int angle_s3 = constrain(values[0], 0, 180);
      int angle_s4 = constrain(values[1], 0, 180);
      int angle_s2 = constrain(values[2], 0, 180);
      int angle_s1 = constrain(values[3], 0, 180);

      // Move servos
      s3.write(angle_s3); // shoulder
      s4.write(angle_s4); // base
      s2.write(angle_s2); // elbow
      s1.write(angle_s1); // gripper

      Serial.printf("S3:%d S4:%d S2:%d S1:%d\n",
                    angle_s3, angle_s4, angle_s2, angle_s1);
    } else {
      Serial.println("Invalid input. Use: s3,s4,s2,s1");
    }
  }
}
