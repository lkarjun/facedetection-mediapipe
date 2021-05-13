import cv2
from mediapipe import solutions
from time import time


class FaceDetection:

    def __init__(self, confidence = .5) -> None:
        self.mpFaceD = solutions.face_detection
        self.face_detection = self.mpFaceD.FaceDetection(0.4)

    def find_faces(self, image, percentage, draw = True):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(imgRGB)

        bbox_ = []
        if results.detections:
            for detection in results.detections:
                bounding_box = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                bbox = int(bounding_box.xmin * w), int(bounding_box.ymin * h), \
                       int(bounding_box.width * w), int(bounding_box.height * h)
                
                bbox_.append([bbox, detection.score])

                if draw: img = self.draw_box(image, bbox, detection.score, percentage=percentage)

        return bbox_, image

    def draw_box(self, image, bbox, detection_score, percentage, l=35, t=5, rt= 0):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(image, bbox, (240, 240, 240), rt)
        # Top Left  x,y
        cv2.line(image, (x, y), (x + l, y), (240, 240, 240), t)
        cv2.line(image, (x, y), (x, y+l), (240, 240, 240), t)
        # Top Right  x1,y
        cv2.line(image, (x1, y), (x1 - l, y), (240, 240, 240), t)
        cv2.line(image, (x1, y), (x1, y+l), (240, 240, 240), t)
        # Bottom Left  x,y1
        cv2.line(image, (x, y1), (x + l, y1), (240, 240, 240), t)
        cv2.line(image, (x, y1), (x, y1 - l), (240, 240, 240), t)
        # Bottom Right  x1,y1
        cv2.line(image, (x1, y1), (x1 - l, y1), (240, 240, 240), t)
        cv2.line(image, (x1, y1), (x1, y1 - l), (240, 240, 240), t)
        if percentage:  cv2.putText(image, f"{int(detection_score[0]*100)}%",
                        (bbox[0], bbox[1] -20), cv2.FONT_HERSHEY_PLAIN,
                         1.5, (240, 240, 240), 2)
        return image

    def start_detection(self, file, wait_key = 0, draw = True, percentage = True):
        cap = cv2.VideoCapture(file)
        pTime = 0
        
        while True:
            success, image = cap.read()
            bbox, image = self.find_faces(image, percentage, draw = draw)
            print(bbox)
            cTime = time()
            fps = 1/(cTime - pTime)
            cv2.imshow("Image", image)
            cv2.waitKey(wait_key)



if __name__ == "__main__":
    FaceDetection(confidence=0.6).start_detection(0, 10, True, False)
