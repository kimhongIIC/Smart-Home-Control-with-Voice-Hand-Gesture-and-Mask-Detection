
import time
from time import sleep
import cv2
import imutils
import numpy as np
import pyfirmata
from pyfirmata import SERVO
import speech_recognition as sr
from imutils.video import VideoStream
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import mediapipe as mp
class HomeAutomation:
    def __init__(self):
        # Check your port in device manager (bluetooth port)
        self.board = pyfirmata.Arduino('COM12')
        # Servo pin
        self.pin = 10
        # led pin
        self.led1_pin = self.board.get_pin('d:7:o')
        self.led2_pin = self.board.get_pin('d:6:o')
        # fan pin
        self.fan_pin1 = self.board.get_pin('d:13:o')
        self.fan_pin2 = self.board.get_pin('d:12:o')
        self.fan_pin3 = self.board.get_pin('d:11:o')
        self.board.digital[self.pin].mode = SERVO
        
        self.faceNet = None
        self.maskNet = None

    def initialize(self):
        time.sleep(2.0)
        self.load_face_detector()
        self.load_mask_detector()

    def load_face_detector(self):
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    def load_mask_detector(self):
        self.maskNet = load_model("model\mask_detector.model")

    def rotate_servo(self, angle):
        self.board.digital[self.pin].write(angle)
        sleep(0.015)

    def detect_and_predict_mask(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        faces = []
        locs = []
        preds = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)
        return (locs, preds)

    def control_home_by_speech(self):
        r = sr.Recognizer()
        mic = sr.Microphone(device_index=1)
        with mic as source:
            r.adjust_for_ambient_noise(source)
            while True:
                audio = r.listen(source)
                try:
                    command = r.recognize_google(audio)
                    print("Heard command:", command)
                    if command == 'Open Door':
                        print('Switch is on')
                        self.rotate_servo(180)
                    elif command == 'close door':
                        for i in range(180, 0, -5):
                            self.rotate_servo(0)
                    elif command == 'LED on':
                        print("LEDs are on")
                        self.led1_pin.write(1)
                    elif command == 'LED off':
                        print("LEDs are off")
                        self.led1_pin.write(0)
                    elif command == 'fan on':
                        self.fan_pin2.write(0)
                        self.fan_pin3.write(1)
                        self.fan_pin1.write(1)
                    elif command == 'fan off':
                        self.fan_pin3.write(0)
                        self.fan_pin2.write(0)
                        self.fan_pin1.write(0)
                    elif command == 'stop':
                        break
                    else:
                        print("Unknown command:", command)
                except sr.UnknownValueError:
                    print("Unknown speech")
                except sr.RequestError as e:
                    print("Error occurred during speech recognition:", str(e))

    def control_home_by_face_and_gesture(self):
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=600)
            (locs, preds) = self.detect_and_predict_mask(frame)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                if mask > withoutMask:
                    label = "Mask"
                else: 
                    label = "No mask"
                if label == "Mask":
                    color = (0,255,0)
                    self.rotate_servo(0)
                else:
                    color = (0,0,255)
                    self.rotate_servo(180)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.rotate_servo(0)
                break
        cv2.destroyAllWindows()
        vs.stop()

    def control_home_by_gesture(self):
        video = cv2.VideoCapture(0)
        mp_draw = mp.solutions.drawing_utils
        mp_hand = mp.solutions.hands
        tipIds = [4, 8, 12, 16, 20]
        with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while True:
                ret, image = video.read()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                lmList = []
                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        myHands = results.multi_hand_landmarks[0]
                        for id, lm in enumerate(myHands.landmark):
                            h, w, c = image.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            lmList.append([id, cx, cy])
                        mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
                fingers = []
                if len(lmList) != 0:
                    if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)
                    total = fingers.count(1)

                    if total == 0:
                        cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, "0", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        cv2.putText(image, "All off", (100, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        self.led1_pin.write(0)
                        self.led2_pin.write(0)
                        self.fan_pin1.write(0)
                        self.fan_pin2.write(0)
                        self.fan_pin3.write(0)
                    elif total == 1:
                        cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, "1", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        cv2.putText(image, "LED on", (100, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        self.led1_pin.write(1)
                        self.led2_pin.write(1)
                        self.fan_pin1.write(0)
                        self.fan_pin2.write(0)
                        self.fan_pin3.write(0)
                    elif total == 2:
                        cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, "2", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        cv2.putText(image, "Fan on", (100, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        self.led1_pin.write(0)
                        self.led2_pin.write(0)
                        self.fan_pin1.write(1)
                        self.fan_pin2.write(0)
                        self.fan_pin3.write(1)
                    elif total == 3:
                        cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, "3", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        cv2.putText(image, "All on", (100, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        self.led1_pin.write(1)
                        self.led2_pin.write(1)
                        self.fan_pin1.write(1)
                        self.fan_pin2.write(0)
                        self.fan_pin3.write(1)
                    elif total == 4:
                        cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, "4", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        cv2.putText(image, "", (100, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                    elif total == 5:
                        cv2.rectangle(image, (20, 300), (270, 425), (0, 255, 0), cv2.FILLED)
                        cv2.putText(image, "5", (45, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                        cv2.putText(image, "", (100, 375), cv2.FONT_HERSHEY_SIMPLEX,
                                    2, (255, 0, 0), 5)
                cv2.imshow("Frame", image)
                k = cv2.waitKey(1)
                if k == ord('q'):
                    self.led1_pin.write(0)
                    self.led2_pin.write(0)
                    self.fan_pin1.write(0)
                    self.fan_pin2.write(0)
                    self.fan_pin3.write(0)
                    break
        video.release()
        cv2.destroyAllWindows()

    def run(self):
        self.initialize()
        print("1. Control home by Speech recognition")
        print("2. Control home by Face recognition and hand gesture")
        option = input("Which one do you prefer: ")
        if option == '1':
            self.control_home_by_speech()
        elif option == '2':
            self.control_home_by_face_and_gesture()
            self.control_home_by_gesture()
        else:
            print("Invalid option")

home_auto = HomeAutomation()
home_auto.run()