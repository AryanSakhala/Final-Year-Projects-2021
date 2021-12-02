
import mediapipe as mp
import cv2
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import time
from math import sqrt
import win32api
import pyautogui
from flask import Flask, render_template, Response
import cv2
app=Flask(__name__)
camera = cv2.VideoCapture(0)

##########################################################################################
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
click=0
 
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

########################################################################################

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
            frame = cv2.flip(frame, 1)
    
            frameHeight, frameWidth, _ = frame.shape
    
            results = hands.process(frame)
    
    
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )
    
            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in mp_hands.HandLandmark:
        
            
                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, frameWidth, frameHeight)
            
                        point=str(point)
        
                        if point=='HandLandmark.INDEX_FINGER_TIP':
                            try:
                                indexfingertip_x=pixelCoordinatesLandmark[0]
                                indexfingertip_y=pixelCoordinatesLandmark[1]
                                win32api.SetCursorPos((indexfingertip_x*4,indexfingertip_y*5))
            
                            except:
                                pass
        
                        elif point=='HandLandmark.THUMB_TIP':
                            try:
                                thumbfingertip_x=pixelCoordinatesLandmark[0]
                                thumbfingertip_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)
            
                            except:
                                pass
        
                        try:
                            #pyautogui.moveTo(indexfingertip_x,indexfingertip_y)
                            Distance_x= sqrt((indexfingertip_x-thumbfingertip_x)**2 + (indexfingertip_x-thumbfingertip_x)**2)
                            Distance_y= sqrt((indexfingertip_y-thumbfingertip_y)**2 + (indexfingertip_y-thumbfingertip_y)**2)
                            if Distance_x<5 or Distance_x<-5:
                                if Distance_y<5 or Distance_y<-5:
                                    click=click+1
                                    if click%5==0:
                                        print("single click")
                                        pyautogui.click()                            
        
                        except:
                            pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)