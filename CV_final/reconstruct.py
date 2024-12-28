# Utility

import cv2
import os
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import tqdm
from PIL import Image


def warping_single(image_r_dir, homography, mask_r_dir, index):
    image_r = cv2.imread(image_r_dir)
    mask_r = cv2.imread(mask_r_dir, 0)
    h, w = image_r.shape[:2]
    mask_r = cv2.resize(mask_r, (w, h), interpolation=cv2.INTER_NEAREST)
    H = homography
    # make a masked image_r for a specific mask
    masked_image_r = cv2.bitwise_and(image_r, image_r, mask=mask_r)
    # use inverse transform to map the image_r to image_t
    warped_image_t = cv2.warpPerspective(masked_image_r, H, (w, h))
    # save the warped image
    # plt.imsave(f'warped_part_{index}.png', warped_image_t, cmap='gray')
    return warped_image_t


def reconstruct_image(model_map, warping_parts, block_size=16):
    num_blocks_h = 135
    num_blocks_w = 240
    model_map = np.reshape(model_map, (num_blocks_h, num_blocks_w))
    h, w = num_blocks_h * block_size, num_blocks_w * block_size
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            part_index = model_map[i, j] - 1
            if part_index == -1:
                continue
            part_block = warping_parts[part_index][i *
                                                   block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            result[i*block_size:(i+1)*block_size, j *
                   block_size:(j+1)*block_size] = part_block
    return result

# Eval


def benchmark(dir_t):
    image_name = [f'./golden/{dir_t}.png']
    predimage_name = [f'./reconstruction/{dir_t}.png']
    txt_name = [f'./prediction/s_{dir_t}.txt']
    # predimage_name = [f'result_{dir_t}_o.png']
    so_img_paths = [os.path.join(name) for name in predimage_name]
    so_txt_paths = [os.path.join(name) for name in txt_name]
    gt_img_paths = [os.path.join(name) for name in image_name]

    psnr = []
    for so_img_path, so_txt_path, gt_img_path in zip(so_img_paths, so_txt_paths, gt_img_paths):

        print('check image... ', so_img_path)

        s = np.array(Image.open(so_img_path).convert('L'))
        g = np.array(Image.open(gt_img_path).convert('L'))
        f = open(so_txt_path, 'r')

        mask = []
        for line in f.readlines():
            mask.append(int(line.strip('\n')))
        f.close()

        mask = np.array(mask).astype(bool)
        assert np.sum(
            mask) == 13000, 'The number of selection blocks should be 13000'

        s = s.reshape(2160//16, 16, 3840//16,
                      16).swapaxes(1, 2).reshape(-1, 16, 16)
        g = g.reshape(2160//16, 16, 3840//16,
                      16).swapaxes(1, 2).reshape(-1, 16, 16)

        s = s[mask]
        g = g[mask]
        assert not (s == g).all(
        ), "The prediction should not be the same as the ground truth"

        mse = np.sum((s-g)**2)/s.size
        psnr.append(10*np.log10(255**2/mse))

    psnr = np.array(psnr)
    avg_psnr = np.sum(psnr) / len(psnr)

    return avg_psnr

# Reconstruct


def load_pkl_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['homographies1'], data['masks_r1_paths'], data['homographies2'], data['masks_r2_paths'], data['blendings']


def load_txt_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    numbers = [int(line.strip()) for line in lines]
    return np.array(numbers)


def predict(target_list, reference1_list, reference2_list):
    image_r_dir1 = f'./golden/{reference1_list}.png'
    image_r_dir2 = f'./golden/{reference2_list}.png'
    model_map_path = f'/model_map/m_{target_list}.txt'
    model_12_path = f'/model_dir/{target_list}.pkl'
    model_map = load_txt_data(model_map_path)
    Hs1, masks_r1_dir, Hs2, masks_r2_dir, weight = load_pkl_data(model_12_path)
    blendings = []
    for a in tqdm.tqdm(range(len(Hs1))):
        warping_part1 = warping_single(
            image_r_dir1, Hs1[a], masks_r1_dir[a], a)
        warping_part2 = warping_single(
            image_r_dir2, Hs2[a], masks_r2_dir[a], a)

        if (a < 4):
            if weight[a] == 0 or weight[a] == 1:
                blending = warping_part1
                mask = np.any(warping_part2 != [0, 0, 0], axis=-1)
                blending[mask] = warping_part2[mask]
            else:
                blending = weight[a]*warping_part1 + \
                    (1-weight[a])*warping_part2
        else:
            blending = weight[a]*warping_part1 + (1-weight[a])*warping_part2
        blending = blending.astype(np.uint8)
        blendings.append(blending)
    reconstructed_image = reconstruct_image(
        model_map, blendings, block_size=16)
    plt.imsave(f'./reconstruction/{target_list}.png',
               reconstructed_image, cmap='gray')
    psnr = benchmark(target_list)
    return psnr

# Main


dir_t = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062',
         '063', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127']

dir_r1 = ['000', '000', '002', '000', '004', '004', '006', '000', '008', '008', '010', '008', '012', '012', '014', '000', '016', '016', '018', '016', '020', '020', '022', '016', '024', '024', '026', '024', '028', '028', '030', '032', '032', '034', '032', '036', '036', '038', '032', '040', '040', '042', '040', '044', '044', '046', '032', '048', '048', '050', '048', '052', '052', '054', '048', '056', '056', '058', '056', '060', '060',
          '062', '064', '064', '066', '064', '068', '068', '070', '064', '072', '072', '074', '072', '076', '076', '078', '064', '080', '080', '082', '080', '084', '084', '086', '080', '088', '088', '090', '088', '092', '092', '094', '096', '096', '098', '096', '100', '100', '102', '096', '104', '104', '106', '104', '108', '108', '110', '096', '112', '112', '114', '112', '116', '116', '118', '112', '120', '120', '122', '120', '124', '124', '126']

dir_r2 = ['002', '004', '004', '008', '006', '008', '008', '016', '010', '012', '012', '016', '014', '016', '016', '032', '018', '020', '020', '024', '022', '024', '024', '032', '026', '028', '028', '032', '030', '032', '032', '034', '036', '036', '040', '038', '040', '040', '048', '042', '044', '044', '048', '046', '048', '048', '064', '050', '052', '052', '056', '054', '056', '056', '064', '058', '060', '060', '064', '062', '064',
          '064', '066', '068', '068', '072', '070', '072', '072', '080', '074', '076', '076', '080', '078', '080', '080', '096', '082', '084', '084', '088', '086', '088', '088', '096', '090', '092', '092', '096', '094', '096', '096', '098', '100', '100', '104', '102', '104', '104', '112', '106', '108', '108', '112', '110', '112', '112', '128', '114', '116', '116', '120', '118', '120', '120', '128', '122', '124', '124', '128', '126', '128', '128']

total_psnr = []
for i in range(len(dir_t)):
    psnr = predict(dir_t[i], dir_r1[i], dir_r2[i])
    print('PSNR: %.5f\n' % (psnr))
    total_psnr.append(psnr)

avg_psnr = np.sum(total_psnr) / len(total_psnr)
print('PSNR: %.5f\n' % (avg_psnr))
