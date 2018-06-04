import cv2
import sys
from HaarClassifier import HaarClassifier

class MLBegineer:
    def __init__(self):
        self.haarClassifier = HaarClassifier()

    def capture(self):
        # użycie filmiku jako żródła obrazu
        # cap = cv2.VideoCapture("MyMovie.mp4")

        # użycie kamery wbudowanej
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            return cap
        else:
            print("Failed on camera capture")
            sys.exit()

    def start(self):
        cap = self.capture()
        while True:
            ret, img = cap.read()
            if ret:
                # konwersja do skali szarości
                grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # wykrywanie za pomocą kaskad Haar'a
                self.haarClassifier.detectCola(grayScale, img)
                self.haarClassifier.detectSprite(grayScale, img)

                # wyświetlenie obrazu
                cv2.imshow('img', img)
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    mLBegineer = MLBegineer()
    mLBegineer.start()
