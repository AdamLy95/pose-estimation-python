from pose_estimator import PoseEstimator
import cv2


model = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
image = cv2.imread("test.jpg")
pose_estimator = PoseEstimator(model)
coords , scores = pose_estimator(image)
for coord in coords:
    p = (int(coord[1]),int(coord[0]))
    image = cv2.circle(image,p,radius = 0,color =(0,255,0),thickness= 10)
cv2.imshow("",image)
cv2.waitKey(0)