import cv2

class HaarClassifier:
    def __init__(self):
        self.colaCascade = cv2.CascadeClassifier('cascades/cola.xml')
        self.spriteCascade = cv2.CascadeClassifier('cascades/sprite.xml')

    def detectCola(self, grayFrame, outputFrame):
        colaScaleFactor = 1.2
        colaMinNeighs = 5
        # wykrycie interesującego nas obszaru za pomocą kaskad
        colas = self.colaCascade.detectMultiScale(grayFrame, colaScaleFactor, colaMinNeighs)
        for (x, y, w, h) in colas:
            # zaznaczanie wykrytego obszaru prostokątem
            cv2.rectangle(outputFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def detectSprite(self, grayFrame, outputFrame):
        spriteScaleFactor = 1.2
        spriteMinNeighs = 5
        # wykrycie interesującego nas obszaru za pomocą kaskad
        sprites = self.spriteCascade.detectMultiScale(grayFrame, spriteScaleFactor, spriteMinNeighs)
        for (x, y, w, h) in sprites:
            # zaznaczanie wykrytego obszaru prostokątem
            cv2.rectangle(outputFrame, (x, y), (x + w, y + h), (0, 0, 255), 2)
