import tensorflow as tf
import numpy as np 
import cv2
from scipy.special import expit

class PoseEstimator():
    def __init__(self,model):
        self.interpreter = tf.lite.Interpreter(model)
        self.interpreter.allocate_tensors()
        self.in_idx = self.interpreter.get_input_details()
        self.out_idx = self.interpreter.get_output_details()

    @staticmethod
    def _im_normalize(img):
        return np.ascontiguousarray(
            2 * ((img / 255) - 0.5
    ).astype('float32'))

    def preprocess_img(self,img):
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
        height,width,n_keypoints = heatmaps.shape[1:]

        maximum_likelihood_keypoints = np.squeeze(heatmaps).reshape(-1,heatmaps.shape[3])
        maximum_likelihood_keypoints = np.array(np.unravel_index(np.argmax(maximum_likelihood_keypoints,axis=0),heatmaps.shape[1:3])).T
        
        offsetVector = np.array([ [offsets[0,y,x,i],offsets[0,y,x,i+n_keypoints]] for i,(y,x) in enumerate(maximum_likelihood_keypoints.astype(int))])
        
        coords =  (maximum_likelihood_keypoints / (np.array([height,width])-1) + offsetVector/257)* np.array([im_h,im_w])        
        confidence_scores = [expit(heatmaps[0,y,x,i])  for i,(y,x) in enumerate(maximum_likelihood_keypoints.astype(int))]
        
        return coords,confidence_scores
    

