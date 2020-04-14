import multi_pose_estimator 
import cv2
import numpy as np
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
---
See connections in multi_estimator
'''

model = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
image = cv2.imread("test2.jpg")
pose_estimator = multi_pose_estimator.PoseEstimator(model)
poses_coords,poses_scores = pose_estimator(cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB))
# poses_coords = [ [person]_1,[person]_2 ,..., [person]_n], Each [person] consist of (17,2) array containing y,x for each keypoint numbered above.
# poses_scores = [ [person]_1,[person]_2 ,..., [person]_n], Each [person] consist of (17,1) array containing score for each keypoint numbered above.

c = np.random.randint(256,size=(len(poses_coords),3))
for i,person in enumerate(poses_coords):
    for f_list in np.array(person):
        coord = f_list
        p = (coord[1],coord[0])
        image = cv2.circle(image,p,radius = 0,color =c[i].tolist(),thickness= 10)
    for connection in multi_pose_estimator.parentChildrenTuples:
        y0, x0 = person[connection[0]]
        y1, x1 = person[connection[1]]
        image = cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), color = c[i].tolist(), thickness=1)


cv2.imwrite("ntest2.jpg",image)
