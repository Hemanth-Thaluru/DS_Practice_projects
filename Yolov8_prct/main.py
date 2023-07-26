import cv2
from ultralytics import YOLO
import numpy as np

model=YOLO("yolov8m.pt")

cap=cv2.VideoCapture("yt.mp4")
while True:
    ret,frame=cap.read()
    if not ret:
        break
    results=model(frame,device="mps")
    result=results[0]
    boxes=np.array(result.boxes.xyxy.cpu(),dtype="int")
    classes=np.array(result.boxes.cls.cpu(),dtype="int")
    for (x,y,x1,y1),cls in zip(boxes,classes):
        cv2.rectangle(frame,(x,y),(x1,y1),(0,0,255),2)
        cv2.putText(frame,result[0].names[cls],(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imshow("Img",frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()