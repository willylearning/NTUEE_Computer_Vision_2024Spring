import numpy as np
import pickle
import math
import os
import re
import cv2

def match_images(path1, path2, dir1, dir2):
    matched_images = {}
    objects = ['sky-other-merged', 'building-other-merged', 'tree-merged', 'road', 'pavement-merged', 'fence-merged', 'car', 'motorcycle', 'person', 'traffic light', 'bus', 'bicycle']
    counts1 = {t: 0 for t in objects}
    counts2 = {t: 0 for t in objects}
    patterns = {t: re.compile(rf'^segment_{t}-\d+\..+$') for t in objects}
    for filename in os.listdir(path1):
        for t, pattern in patterns.items():
            if pattern.match(filename):
                counts1[t] += 1
    for filename in os.listdir(path2):
        for t, pattern in patterns.items():
            if pattern.match(filename):
                counts2[t] += 1
    for t in objects:
        if counts1[t] == 1 and counts2[t] == 1:
            print(f'{dir1}_segment_{t}-0 == {dir2}_segment_{t}-0')
            matched_images[f'segment_{t}-0'] = f'segment_{t}-0'
        elif counts1[t] == 0 or counts2[t] == 0:
            print(f'{t} False')
        else:
            num1 = counts1[t]
            num2 = counts2[t]
            list1 = [0]*num1
            list2 = [0]*num2
            for i in range(num1):
                img = cv2.imread(f'{path1}segment_{t}-{i}.png', cv2.IMREAD_GRAYSCALE)
                area = np.sum(img == 255)
                moments = cv2.moments(img, binaryImage=True)
                if moments['m00'] != 0:
                    centroid_x = moments['m10'] / moments['m00']
                    centroid_y = moments['m01'] / moments['m00']
                else:
                    centroid_x, centroid_y = 0, 0
                list1[i] = [i, area, centroid_x, centroid_y]
            for i in range(num2):
                img = cv2.imread(f'{path2}segment_{t}-{i}.png', cv2.IMREAD_GRAYSCALE)
                area = np.sum(img == 255)
                moments = cv2.moments(img, binaryImage=True)
                if moments['m00'] != 0:
                    centroid_x = moments['m10'] / moments['m00']
                    centroid_y = moments['m01'] / moments['m00']
                else:
                    centroid_x, centroid_y = 0, 0
                list2[i] = [i, area, centroid_x, centroid_y]
            sorted_lst1 = sorted(list1, key=lambda x: x[1], reverse=True)
            sorted_lst2 = sorted(list2, key=lambda x: x[1], reverse=True)
            matched_indices = {}
            for i in range(num1):
                for j in range(num2):
                    if i not in matched_indices and j not in matched_indices.values():
                        distance = math.sqrt((sorted_lst1[i][2] - sorted_lst2[j][2]) ** 2 + (sorted_lst1[i][3] - sorted_lst2[j][3]) ** 2)
                        if distance < 30 and abs(sorted_lst1[i][1] - sorted_lst2[j][1])<sorted_lst1[i][1]*0.5:
                            matched_images[f'segment_{t}-{sorted_lst1[i][0]}'] = f'segment_{t}-{sorted_lst2[j][0]}'
                            print(f'{dir1}_segment_{t}-{sorted_lst1[i][0]} == {dir2}_segment_{t}-{sorted_lst2[j][0]}')
                            matched_indices[i] = j
    return matched_images

def find_feature(mask1_dir, mask2_dir, image1_dir, image2_dir):
    image1 = cv2.imread(image1_dir)
    image2 = cv2.imread(image2_dir)
    mask1 = cv2.imread(mask1_dir, 0)
    mask2 = cv2.imread(mask2_dir, 0)
    masked_image1 = apply_mask(image1, mask1)
    keypoints1, descriptors1 = detect_and_compute(masked_image1)

    masked_image2 = apply_mask(image2, mask2)
    keypoints2, descriptors2 = detect_and_compute(masked_image2)
    return keypoints1, keypoints2, descriptors1, descriptors2

def apply_mask(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(image, image, mask=mask)

def detect_and_compute(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def find_homography_with_ransac(mask1_dir, mask2_dir, keypoints1, keypoints2, descriptors1, descriptors2):
    mask1 = cv2.imread(mask1_dir, 0)
    mask2 = cv2.imread(mask2_dir, 0)
    good_matches = []
    index_params = dict(algorithm=1, trees=10)  # trees count can be adjusted
    search_params = dict(checks=100)  # checks can be adjusted
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # 0.6 can be adjusted
            good_matches.append(m)
    if len(good_matches) > 4:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, mask=(mask1 & mask2))
        return H, status
    else:
        return None, None

def save_data(homographies1, homographies2, masks_r1_paths, masks_r2_paths, masks_t1_paths, masks_t2_paths, filename):
        data = {
            'homographies1': homographies1,
            'masks_r1_paths': masks_r1_paths,
            'masks_t1_paths': masks_t1_paths,
            'homographies2': homographies2,
            'masks_r2_paths': masks_r2_paths,
            'masks_t2_paths': masks_t2_paths
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

target_list =     ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127']
reference1_list = ['000', '000', '002', '000', '004', '004', '006', '000', '008', '008', '010', '008', '012', '012', '014', '000', '016', '016', '018', '016', '020', '020', '022', '016', '024', '024', '026', '024', '028', '028', '030', '032', '032', '034', '032', '036', '036', '038', '032', '040', '040', '042', '040', '044', '044', '046', '032', '048', '048', '050', '048', '052', '052', '054', '048', '056', '056', '058', '056', '060', '060', '062', '064', '064', '066', '064', '068', '068', '070', '064', '072', '072', '074', '072', '076', '076', '078', '064', '080', '080', '082', '080', '084', '084', '086', '080', '088', '088', '090', '088', '092', '092', '094', '096', '096', '098', '096', '100', '100', '102', '096', '104', '104', '106', '104', '108', '108', '110', '096', '112', '112', '114', '112', '116', '116', '118', '112', '120', '120', '122', '120', '124', '124', '126']
reference2_list = ['002', '004', '004', '008', '006', '008', '008', '016', '010', '012', '012', '016', '014', '016', '016', '032', '018', '020', '020', '024', '022', '024', '024', '032', '026', '028', '028', '032', '030', '032', '032', '034', '036', '036', '040', '038', '040', '040', '048', '042', '044', '044', '048', '046', '048', '048', '064', '050', '052', '052', '056', '054', '056', '056', '064', '058', '060', '060', '064', '062', '064', '064', '066', '068', '068', '072', '070', '072', '072', '080', '074', '076', '076', '080', '078', '080', '080', '096', '082', '084', '084', '088', '086', '088', '088', '096', '090', '092', '092', '096', '094', '096', '096', '098', '100', '100', '104', '102', '104', '104', '112', '106', '108', '108', '112', '110', '112', '112', '128', '114', '116', '116', '120', '118', '120', '120', '128', '122', '124', '124', '128', '126', '128', '128']


if not os.path.exists('homography'):
    os.makedirs('homography')

for i in range(len(target_list)):

    dir_t = target_list[i]  # which is your target image num
    dir_r1 = reference1_list[i]  # which is your reference1 image num
    dir_r2 = reference2_list[i]  # which is your reference2 image num
    path_t = f'./segmentation/{dir_t}.png/'  # where to read target mask
    path_r1 = f'./segmentation/{dir_r1}.png/'  # where to read reference1 mask
    path_r2 = f'./segmentation/{dir_r2}.png/'  # where to read reference2 mask
    path_golden = './golden/'
    save_path = './homography/'

    match_masks1 = {}
    match_masks1 = match_images(path_t, path_r1, dir_t, dir_r1)
    match_masks2 = {}
    match_masks2 = match_images(path_t, path_r2, dir_t, dir_r2)
    Hs1 = []
    Hs2 = []
    masks_t1_dir = []
    masks_t2_dir = []
    masks_r1_dir = []
    masks_r2_dir = []
    for key, value in match_masks1.items():
        mask_t_dir = path_t + key + '.png'
        mask_r_dir = path_r1 + value + '.png'
        image_t_dir = path_golden + dir_t + '.png'
        image_r_dir = path_golden + dir_r1 + '.png'
        keypoints_t, keypoints_r, descriptors_t, descriptors_r = find_feature(mask_t_dir, mask_r_dir, image_t_dir, image_r_dir)
        H, inliners = find_homography_with_ransac(mask_t_dir, mask_r_dir ,keypoints_t, keypoints_r, descriptors_t, descriptors_r)
        if H is not None:
            Hs1.append(H)
            masks_t1_dir.append(mask_t_dir)
            masks_r1_dir.append(mask_r_dir)
    for key, value in match_masks2.items():
        mask_t_dir = path_t + key + '.png'
        mask_r_dir = path_r2 + value + '.png'
        image_t_dir = path_golden + dir_t + '.png'
        image_r_dir = path_golden + dir_r2 + '.png'
        keypoints_t, keypoints_r, descriptors_t, descriptors_r = find_feature(mask_t_dir, mask_r_dir, image_t_dir, image_r_dir)
        H, inliners = find_homography_with_ransac(mask_t_dir, mask_r_dir ,keypoints_t, keypoints_r, descriptors_t, descriptors_r)
        if H is not None:
            Hs2.append(H)
            masks_t2_dir.append(mask_t_dir)
            masks_r2_dir.append(mask_r_dir)

    data_file = f'{save_path}data_{dir_t}to{dir_r1}_{dir_r2}.pkl'
    save_data(Hs1, Hs2, masks_r1_dir, masks_r2_dir, masks_t1_dir, masks_t2_dir, data_file)