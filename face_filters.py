import cv2
import dlib
from imutils import face_utils

face_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def detect(img, cascade, minimumFeatureSize=(80, 80)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]  # convert last coord from (width,height) to (maxX, maxY)
    return rects


def feats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    te = detect(gray, face_cascade, (80, 80))
    if len(te) == 0:
        return
    if len(te) > 1:
        face = te[0]
    if len(te) == 1:
        [face] = te
    # cv2.rectangle(frame, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
    rect = dlib.rectangle(left=int(face[0]), top=int(face[1]), right=int(face[2]), bottom=int(face[3]))
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    for i in range(68):
        cv2.circle(frame, (shape[i][0], shape[i][1]), 2, (0, 0, 255), -1)


def main():
    cam = cv2.VideoCapture(0)
    while True:
        r, frame = cam.read()
        if not r:
            continue
        feats(frame)
        cv2.imshow("you", frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
    cam.release()


if __name__ == '__main__':
    main()
