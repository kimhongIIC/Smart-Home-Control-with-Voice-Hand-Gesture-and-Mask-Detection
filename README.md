# Smart Home Control with Voice, Hand Gesture, and Mask Detection

This project demonstrates a smart home prototype integrating multiple human-computer interaction methods to control home devices such as LEDs, fans, and a door servo motor. Users can interact via:

1. **Speech Recognition** – Voice commands to control doors, lights, and fans.
2. **Hand Gestures** – Using a webcam and gesture recognition to toggle devices.
3. **Mask Detection** – A face mask detection model that automatically adjusts a door servo based on whether a user is wearing a mask or not.

All device control is facilitated through an Arduino Uno, using the PyFirmata library for direct communication and control of the servo motor, LEDs, and fan.

---

## Features

- **Voice Commands:**  
  Use spoken commands:
  - "Open Door" or "close door" to control the servo-powered door.
  - "LED on" or "LED off" to switch LEDs.
  - "fan on" or "fan off" to operate the fan.
  - "stop" to exit speech mode.

- **Mask Detection for Door Control:**  
  A camera feed checks if a person is wearing a mask.  
  - If mask is detected: Servo stays at 0° (closed door).
  - If no mask detected: Servo moves to 180° (open door).

- **Hand Gesture Control:**  
  Using a webcam, show a number of fingers to control devices:
  - 0 fingers: Turn all devices off
  - 1 finger: Turn LEDs on
  - 2 fingers: Turn the fan on
  - 3 fingers: Turn all on (LEDs + Fan)

- **Arduino Integration:**  
  Uses PyFirmata to control:
  - LEDs connected to digital pins
  - Fan connected to digital pins
  - Servo motor on pin 10 for door control

---

## Requirements

**Hardware:**
- Arduino Uno or compatible board
- Servo motor (connected to pin 10)
- LEDs and appropriate resistors (connected to pins 6 and 7)
- DC fan or similar device (connected to pins 11, 12, and 13)
- USB cable to connect Arduino
- Webcam for mask detection and gesture recognition
- Microphone for speech recognition

**Software & Libraries:**
- Python 3.x  
- [PyFirmata](https://pypi.org/project/pyfirmata/) (`pip install pyfirmata`)
- [OpenCV](https://pypi.org/project/opencv-python/) (`pip install opencv-python`)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) (`pip install SpeechRecognition`)
- [Mediapipe](https://pypi.org/project/mediapipe/) (`pip install mediapipe`)
- [TensorFlow](https://pypi.org/project/tensorflow/) (`pip install tensorflow`)
- [Imutils](https://pypi.org/project/imutils/) (`pip install imutils`)

**Models & Files:**
- Face detector: `face_detector/deploy.prototxt` and `face_detector/res10_300x300_ssd_iter_140000.caffemodel`
- Mask detector model: `model/mask_detector.model`

Update the file paths in the code if your directory structure differs.

---

## Setup Instructions

1. **Hardware Setup:**
   - Connect the Arduino Uno to the computer.
   - Connect LEDs, servo, and fan as per the code’s pin definitions.
   - Verify servo on pin 10, LEDs on pins 6 & 7, and fan on pins 11, 12, 13.

2. **Arduino & PyFirmata:**
   - Update the Arduino COM port in the code:
     ```python
     self.board = pyfirmata.Arduino('COM12')
     ```
     Change `'COM12'` to your actual COM port (e.g. `'COM3'` or `'/dev/ttyACM0'`).

3. **Model Files:**
   - Place the face detector files in `face_detector` directory.
   - Place `mask_detector.model` in the `model` directory.

4. **Run the Program:**
   ```bash
   python home.py

Choose how you want to control the system:

- 1: Control by Speech recognition
- 2: Control by Face recognition (mask detection) and hand gesture

## Usage
### Speech Recognition Mode:

Say "Open Door" to open, "close door" to close.
Say "LED on"/"LED off" to toggle LEDs.
Say "fan on"/"fan off" to toggle the fan.
Say "stop" to exit speech mode.
Face Recognition & Gesture Mode:

### With a mask, door stays closed (servo 0°).
### Without a mask, door opens (servo 180°).
### Hand gestures:
0 fingers: All off
1 finger: LEDs on
2 fingers: Fan on
3 fingers: All on (LEDs + Fan)
Press 'q' to exit camera-based modes.

## Customization
- Pin Assignments: Modify pin definitions in the code.
- Voice Commands: Update the control_home_by_speech() method for new commands.
- Gesture Mappings: Adjust logic in control_home_by_gesture() for different actions.