import tensorflow as tf
import numpy as np 
import cv2
from scipy.special import expit
from scipy.ndimage import maximum_filter
from queue import PriorityQueue
class PoseEstimator():
    def __init__(self,model):
        self.interpreter = tf.lite.Interpreter(model)
        self.interpreter.allocate_tensors()
        self.in_idx = self.interpreter.get_input_details()
        self.out_idx = self.interpreter.get_output_details()
        self.p_queue = PriorityQueue()

    @staticmethod
    def _im_normalize(img):
        return np.ascontiguousarray(
            2 * ((img / 255) - 0.5
    ).astype('float32'))

    def preprocess_img(self,img):
        # fit the image into a 257x257 square
        img_small = cv2.resize(img, (257, 257))
        img_small = np.ascontiguousarray(img_small)
        img_norm = self._im_normalize(img_small)
        return img_norm


    def __call__(self,image):
        im_h,im_w = image.shape[:2]
        img_norm = self.preprocess_img(image)
        img_norm = np.expand_dims(img_norm, axis=0)
        self.interpreter.set_tensor(self.in_idx[0]['index'], img_norm)
        self.interpreter.invoke()

        heatmaps = self.interpreter.get_tensor(self.out_idx[0]['index'])
        offsets = self.interpreter.get_tensor(self.out_idx[1]['index'])
        displacementForward = self.interpreter.get_tensor(self.out_idx[2]['index'])
        displacementBackward = self.interpreter.get_tensor(self.out_idx[3]['index'])   

        p_q = self._get_local_minima(heatmaps.copy(),offsets.copy())
        poses_scores,poses_coords = self._connecting_people(p_q,heatmaps,displacementBackward,displacementForward,offsets)
        poses_coords =  np.round(poses_coords*np.array([im_h,im_w])/257).astype(int)
        poses_scores = expit(poses_scores)
        return poses_coords,poses_scores
    

    def _get_local_minima(self,heatmaps,offsets):
        p_queue = PriorityQueue()
        score_threshold = 1e-2 # Taken from the paper, is it score_threshold for h or expit(h), I do not know yet
        n_keypoints = heatmaps.shape[-1]
        for keypoint in range(n_keypoints):
            h = heatmaps[...,keypoint]
            h[h < score_threshold] =0
            maximum_value = maximum_filter(h,size=3) # Hyper-parameter size, set to 3 by trial-n-error
            res = (maximum_value >0) & (h > 0)
            res = np.array(res.nonzero()).T
            [p_queue.put((
                -h[r[0],r[1],r[2]], #Priority, for some reason standard Python priority queue is increasing
                np.array(r[1:]*257/8 + offsets[0,int(r[1]),int(r[2]),[keypoint,keypoint+n_keypoints]]),
                keypoint,
            )) for r in res]
        return p_queue

    def _check_radius(self,p1,p2,threshold=32):
        return np.linalg.norm([p1-p2]) > threshold

    def _connecting_people(self,p_queue,scores,displacementBackward,displacementForward,offsets):
        poses_scores,poses_coords = [],[]
        while not p_queue.empty():
            root = p_queue.get()
            if np.all([self._check_radius(root[1],p[root[-1]]) for p in poses_coords]):
                pose_scores,pose_coords = self._decode_pose(root,scores,displacementBackward,displacementForward,offsets)
                poses_scores.append(pose_scores)
                poses_coords.append(pose_coords)
        return poses_scores,poses_coords 
    
    def _decode_pose(self,root,scores,displacementBackward,displacementForward,offsets):
        pose_scores,pose_coords = np.zeros(17),np.zeros((17,2))
        pose_scores[root[-1]] = -root[0] # Priority was reversed because the Python priority queue is increasing
        pose_coords[root[-1]] = root[1]

        for e in reversed(range(16)):
            target_edge_id,source_edge_id = parentChildrenTuples[e]
            if pose_scores[source_edge_id] != 0 and pose_scores[target_edge_id] == 0:
                pose_scores[target_edge_id],pose_coords[target_edge_id] = self._traverse_to_keypoint(pose_coords[source_edge_id],target_edge_id,scores,displacementBackward[...,[e,e+16]],offsets)
        for e in range(16):
            source_edge_id , target_edge_id = parentChildrenTuples[e]
            if pose_scores[source_edge_id] != 0 and pose_scores[target_edge_id] == 0:
                pose_scores[target_edge_id] , pose_coords[target_edge_id] = self._traverse_to_keypoint(pose_coords[source_edge_id],target_edge_id,scores,displacementForward[...,[e,e+16]],offsets)
        return pose_scores,pose_coords

    def _traverse_to_keypoint(self,source_coords,target_edge,scores,displacement, offsets):
        source_indices = np.round(source_coords *8/257).astype(np.int32)
        target_point = source_coords + displacement[0,source_indices[0], source_indices[1]]

        target_point_indices = np.round(target_point *8/257).astype(np.int32)
        target_point_indices[target_point_indices < 0] = 0
        target_point_indices[target_point_indices > 8] = 8
        
        score = scores[0,target_point_indices[0], target_point_indices[1], target_edge]
        coords = target_point_indices * 257/8 + offsets[0,target_point_indices[0], target_point_indices[1], [target_edge,target_edge+17]]

        return score, coords

#TRANSLATED FROM https://github.com/tensorflow/tfjs-models/blob/b72c10bdbdec6b04a13f780180ed904736fa52a5/body-pix/src/keypoints.ts
PART_NAMES = [
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
]
NUM_KEYPOINTS = len(PART_NAMES)

PART_IDS =  {k: i for i,k in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES=[
  ['leftHip', 'leftShoulder'], ['leftElbow', 'leftShoulder'],
  ['leftElbow', 'leftWrist'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['rightHip', 'rightShoulder'],
  ['rightElbow', 'rightShoulder'], ['rightElbow', 'rightWrist'],
  ['rightHip', 'rightKnee'], ['rightKnee', 'rightAnkle'],
  ['leftShoulder', 'rightShoulder'], ['leftHip', 'rightHip']
]

# /*
#  * Define the skeleton. This defines the parent->child relationships of our
#  * tree. Arbitrarily this defines the nose as the root of the tree, however
#  * since we will infer the displacement for both parent->child and
#  * child->parent, we can define the tree root as any node.
#  */
POSE_CHAIN =[
  ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
  ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
  ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
  ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
  ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
  ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
  ['rightKnee', 'rightAnkle']
]
parentChildrenTuples = [ (PART_IDS[x],PART_IDS[y]) for x,y in POSE_CHAIN]
 