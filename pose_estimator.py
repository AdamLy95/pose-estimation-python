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
        # fit the image into a 257x257 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 2
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        img_small = cv2.resize(img_pad, (257, 257))
        img_small = np.ascontiguousarray(img_small)
        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad


    def __call__(self,image):
        im_h,im_w = image.shape[:2]
        img_pad, img_norm, pad  = self.preprocess_img(image)
        img_norm = np.expand_dims(img_norm, axis=0)
        self.interpreter.set_tensor(self.in_idx[0]['index'], img_norm)
        self.interpreter.invoke()

        heatmaps = self.interpreter.get_tensor(self.out_idx[0]['index'])
        offsets = self.interpreter.get_tensor(self.out_idx[1]['index'])
        height,width,n_keypoints = heatmaps.shape[1:]
        maximum_likelihood_keypoints = np.zeros((n_keypoints,2))

        for i in range(n_keypoints):
            h_map = heatmaps[...,i]
            maximum_likelihood_keypoints[i,:] = np.unravel_index(np.argmax(h_map), h_map.shape)[1:]
        confidence_scores = np.zeros(n_keypoints)
        coords = np.zeros((n_keypoints,2))

        for i,p in enumerate(maximum_likelihood_keypoints):
            y,x = p.astype(int)
            coords[i,0] = p[0] / (height - 1) * im_h + offsets[0,y,x,i] * im_h/257
            coords[i,1] = p[1] / (width - 1) * im_h + offsets[0,y,x,i+n_keypoints] * im_w/257
            confidence_scores[i] = expit(heatmaps[0,y,x,i])
        
        return coords,confidence_scores
    

