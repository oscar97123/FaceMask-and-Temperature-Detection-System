#include <Adafruit_MLX90614.h>
#define echoPin 2 // attach pin D2 Arduino to pin Echo of HC-SR04
#define trigPin 3

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

long duration; // variable for the duration of sound wave travel
int distance;


void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT); // Sets the trigPin as an OUTPUT
  pinMode(echoPin, INPUT); // Sets the echoPin as an INPUT

  pinMode(13, OUTPUT); 
 
//  Serial.println("Adafruit MLX90614 test");

  if (!mlx.begin()) {
//    Serial.println("Error connecting to MLX sensor. Check wiring.");
  };

  digitalWrite(13, HIGH); 
}

void loop() {
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    // Sets the trigPin HIGH (ACTIVE) for 10 microseconds
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    // Reads the echoPin, returns the sound wave travel time in microseconds
    duration = pulseIn(echoPin, HIGH);
    // Calculating the distance
    distance = duration * 0.034 / 2;
    if(distance < 50){
      Serial.println(mlx.readObjectTempC()); 
      delay(1000);
    }
  }
