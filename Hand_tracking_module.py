import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import time


class handDetector():
     def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
          self.mode = mode
          self.maxHands = maxHands
          self.modelComplex = modelComplexity
          self.detectionCon = detectionCon
          self.trackCon = trackCon
          

          self.mphands = mp.solutions.hands
          self.hands = self.mphands.Hands(self.mode, self.maxHands,self.modelComplex,
                                           self.detectionCon,self.trackCon)
          self.mpDraw = mp.solutions.drawing_utils


     def findHands(self, img, draw=True):
          img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          self.results = self.hands.process(img_RGB)

         # print(results.multi_hand_landmarks)
          if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw: 
                   self.mpDraw.draw_landmarks(img, handlms, self.mphands.HAND_CONNECTIONS)
          return img
              
     
     def findPositiion(self, img, handNo=0, draw=True):

          lmList = []
          if self.results.multi_hand_landmarks:
               myHand = self.results.multi_hand_landmarks[handNo]

               for id, lm in enumerate(myHand.landmark):
                    #print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
                    if draw:
                         cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)
          return lmList            
                      

   
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success , img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPositiion(img)

        if len(lmList) !=0:
             print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
                 (255,0,255), 3)
                
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
              break
                
       


if __name__ == "__main__":
      main()