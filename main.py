from pose_estimator import PoseEstimator
import cv2


model = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
image = cv2.imread("test.jpg")
pose_estimator = PoseEstimator(model)
coords , scores = pose_estimator(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB))
'''
0 	nose
1 	leftEye
2 	rightEye
3 	leftEar
4 	rightEar
5 	leftShoulder
6 	rightShoulder
7 	leftElbow
8 	rightElbow
9 	leftWrist
10 	rightWrist
11 	leftHip
12 	rightHip
13 	leftKnee
14 	rightKnee
15 	leftAnkle
16 	rightAnkle
'''
for coord in coords:
    p = (int(coord[1]),int(coord[0]))
    image = cv2.circle(image,p,radius = 0,color =(0,255,0),thickness= 10)
cv2.imwrite("ntest.jpg",image)
