import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def ransac(src_points, dst_points, threshold = 0.5, num_iters = 3000, k = 4):
    num_best_inliers = 0
    best_H = None
    for i in range(num_iters):
        # get the index of ramdom sampling (randomly pick k(>=4) points for computing homography)
        idx = random.sample(range(len(src_points)), k)
        # solve homography
        H = solve_homography(src_points[idx], dst_points[idx])
        # calculate the coordination after transform to dst img
        U = np.vstack((src_points.T, np.ones((1, src_points.shape[0]))))
        dst_coor = np.dot(H, U) # 3 x 100
        dst_coor = ((dst_coor/dst_coor[2]).T) # 100 x 3
        # compute errors
        errors = np.linalg.norm(dst_coor[:,:-1]-dst_points, axis=1)
        # if errors < threshold => inliers
        num_inliers = np.sum(errors < threshold)
        # find num_best_inliers by repeating num_iters times
        if num_inliers > num_best_inliers :
            best_H = H.copy()
            num_best_inliers = num_inliers    
    return best_H

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx] # queryImage
        im2 = imgs[idx + 1] # trainImage

        # TODO: 1.feature detection & matching
        # initiate ORB detector, extract keypoints
        orb = cv2.ORB_create() 
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        # feature matching (using opencv brute force matcher)
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # match descriptors
        matches = bf.match(des1, des2)
        # sort them in the order of their distance, so that best matches (with low distance) come to front.
        matches = sorted(matches, key = lambda x : x.distance)
        # draw first 100 matches.
        good_matches = matches[:100]
        # use .queryIdx and .trainIdx of DMatch object (bf.match(des1,des2) line is a list of DMatch objects)
        query_idx, train_idx = [], []
        for match in good_matches:
            query_idx.append(match.queryIdx)
            train_idx.append(match.trainIdx)
        dst_points = np.array([kp1[idx].pt for idx in query_idx])
        src_points = np.array([kp2[idx].pt for idx in train_idx]) 

        # TODO: 2. apply RANSAC to choose best H
        best_H = ransac(src_points, dst_points, threshold = 0.5, num_iters = 3000, k = 4)

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)