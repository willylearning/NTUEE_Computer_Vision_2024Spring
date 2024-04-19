import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        # gaussian_images = []
        octave1_images = [image]
        octave2_images = []
        for i in range(1, self.num_guassian_images_per_octave): # i=1~4
            octave1_images.append(cv2.GaussianBlur(image, (0, 0), self.sigma**(i)))
        # print(len(octave1_images))
        # print(len(octave2_images))

        # octave2 (downsampling 1/2)
        octave2_images.append(cv2.resize(octave1_images[-1], None, fx = 0.5, fy = 0.5 ,interpolation=cv2.INTER_NEAREST)) 
        # print(octave2_images[0])
        for j in range(1, self.num_guassian_images_per_octave): # j=1~4
            octave2_images.append(cv2.GaussianBlur(octave2_images[0], (0, 0), self.sigma**(j)))
        # print(octave2_images)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        # dog_images = []
        dog1_images = []
        dog2_images = []
        for i in range(1, self.num_guassian_images_per_octave): # i=1~4
            dog1_images.append(cv2.subtract(octave1_images[i], octave1_images[i-1]))
        # print(dog1_images)
        for j in range(1, self.num_guassian_images_per_octave): # j=1~4
            dog2_images.append(cv2.subtract(octave2_images[j], octave2_images[j-1]))

        # 3D array (4*h*w)
        dog1_images = np.array(dog1_images)
        dog2_images = np.array(dog2_images) 
        # print(dog2_images)
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        h1 = dog1_images.shape[1] # original images size : h1*w1 
        w1 = dog1_images.shape[2]
        # print(dog1_images.shape[0])
        for i in range(1, dog1_images.shape[0]-1): # i=1~2
            for j in range(1, h1-1): # j=1~h1-2
                for k in range(1, w1-1): # k=1~w1-2
                    # find whether the center point is local extremum and use np.abs() to threshold
                    # Padding is no needed : center point won't be on the edge
                    local_min = np.min(dog1_images[i-1:i+2, j-1:j+2, k-1:k+2])
                    local_max = np.max(dog1_images[i-1:i+2, j-1:j+2, k-1:k+2])
                    if((dog1_images[i, j, k] == local_min) or (dog1_images[i, j, k] == local_max)):
                        if(np.abs(dog1_images[i, j, k]) > self.threshold):
                            keypoints.append([j, k])

        h2 = dog2_images.shape[1] # images size : h2*w2 
        w2 = dog2_images.shape[2]
        for i in range(1, dog2_images.shape[0]-1): # i=1~2
            for j in range(1, h2-1): # j=1~h1-2
                for k in range(1, w2-1): # k=1~w1-2
                    # find whether the center point is local extremum and use np.abs() to threshold
                    # Padding is no needed : center point won't be on the edge
                    local_min = np.min(dog2_images[i-1:i+2, j-1:j+2, k-1:k+2])
                    local_max = np.max(dog2_images[i-1:i+2, j-1:j+2, k-1:k+2])
                    if((dog2_images[i, j, k] == local_min) or (dog2_images[i, j, k] == local_max)):
                        if(np.abs(dog2_images[i, j, k]) > self.threshold):
                            keypoints.append([2*j, 2*k])  # *2 here because of octave2 resizing to 1/2       
                    

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)
        # print(keypoints)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
