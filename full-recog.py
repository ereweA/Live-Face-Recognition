import cv2
import threading
from deepface import DeepFace 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

cap = cv2.VideoCapture(0) # use default camera

previous_x, previous_y, previous_w, previous_h = 0, 0, 0, 0  
alpha = 0.1  # smoothing factor

counter = 0

reference_img = cv2.imread("reference.jpg") # load ref img

face_match = False

def check_face(frame):
    global face_match 
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']: # verify current frame and copy of reference image
            face_match = True
        else:
            face_match = False    
    except ValueError:
        face_match = False

while True:
    ret, frame = cap.read() 
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        x = int(alpha * x + (1 - alpha) * previous_x)
        y = int(alpha * y + (1 - alpha) * previous_y)
        w = int(alpha * w + (1 - alpha) * previous_w)
        h = int(alpha * h + (1 - alpha) * previous_h)

        previous_x, previous_y, previous_w, previous_h = x, y, w, h

    if ret: 
        if counter % 30 == 0: 
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start() 
            except ValueError: 
                pass
        counter += 1

        if face_match:
            cv2.rectangle(frame, (previous_x, previous_y), (previous_x + previous_w, previous_y + previous_h), (0, 255, 0), 2)
            cv2.putText(frame, "MATCH", (previous_x, previous_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (previous_x, previous_y), (previous_x + previous_w, previous_y + previous_h), (0, 0, 255), 2)
            cv2.putText(frame, "NO MATCH", (previous_x, previous_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    cv2.imshow('Live Face Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()