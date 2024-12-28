import cv2
import os
import re
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import tqdm
from PIL import Image


# Utility

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


# Selection map

def get_blocks(image, block_size):
    h, w = image.shape[:2]
    blocks = []
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = image[y:y+block_size, x:x+block_size]
            blocks.append(block)
    return blocks


def compute_block_differences(blocks1, blocks2):
    differences = []
    for b1, b2 in zip(blocks1, blocks2):
        diff = np.sum(np.abs(b1.astype(int) - b2.astype(int)))
        differences.append(diff)
    return differences


def selection_map(dir_t):
    # Load images
    # enter the predicted img here
    img1 = cv2.imread(f'./prediction/{dir_t}.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f'./golden/{dir_t}.png', cv2.IMREAD_GRAYSCALE)  # gt
    block_size = 16
    num_blocks_to_select = 13000

    # Split images into blocks
    blocks1 = get_blocks(img1, block_size)
    blocks2 = get_blocks(img2, block_size)

    # Compute differences for each block
    differences = compute_block_differences(blocks1, blocks2)

    # Get indices of the blocks with the smallest differences
    sorted_indices = np.argsort(differences)
    selected_indices = set(sorted_indices[:num_blocks_to_select])

    # Prepare the output list
    output_list = []
    num_blocks_row = img1.shape[0] // block_size
    num_blocks_col = img1.shape[1] // block_size

    # Ensure we process all blocks
    index = 0
    for row in range(num_blocks_row):
        for col in range(num_blocks_col):
            if index in selected_indices:
                output_list.append("1")
            else:
                output_list.append("0")
            index += 1
    # Ensure the output list has exactly 32400 elements
    assert len(output_list) == num_blocks_row * \
        num_blocks_col, f"The output list length is incorrect: {
            len(output_list)}"


# Eval

def benchmark(dir_t):
    image_name = [f'./golden/{dir_t}.png']
    txt_name = [f'./prediction/s_{dir_t}.txt']
    predimage_name = [f'./result_{dir_t}_o.png']
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


# Execute

def execute(dir_t, dir_r1, dir_r2):
    Hs1 = []
    masks_t_dir1 = []
    masks_r_dir1 = []
    Hs2 = []
    masks_t_dir2 = []
    masks_r_dir2 = []

    data_file1 = f'./homography/data_{dir_t}to{dir_r1}_{dir_r2}.pkl'
    image_t_dir = f'./golden/{dir_t}.png'
    image_r_dir1 = f'./golden/{dir_r1}.png'
    image_r_dir2 = f'./golden/{dir_r2}.png'

    golden = cv2.imread(image_t_dir)
    h, w = golden.shape[:2]

    def load_data(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data['homographies1'], data['masks_r1_paths'], data['masks_t1_paths'], data['homographies2'], data['masks_r2_paths'], data['masks_t2_paths']
    # load the data
    Hs1, masks_r_dir1, masks_t_dir1, Hs2, masks_r_dir2, masks_t_dir2 = load_data(
        data_file1)

    result_image1 = np.zeros((h, w, 3), dtype=np.uint8)
    result_image2 = np.zeros((h, w, 3), dtype=np.uint8)

    a = np.zeros((2160//16, 3840//16))  # occupied region
    cnt = 0
    H1_list = []
    H2_list = []
    blendings_list = []
    masks_r_dir1_list = []
    masks_r_dir2_list = []

    for i in range(4):  # max(len(Hs1), len(Hs2))
        # print(f'{masks_t_dir[i]} -> {masks_r_dir[i]}: {Hs[i]}')
        H1_list.append(Hs1[i])
        H2_list.append(Hs2[i])
        masks_r_dir1_list.append(masks_r_dir1[i])
        masks_r_dir2_list.append(masks_r_dir2[i])
        warping_part1 = warping_single(
            image_r_dir1, Hs1[i], masks_r_dir1[i], i)
        warping_part2 = warping_single(
            image_r_dir2, Hs2[i], masks_r_dir2[i], i)
        # combine the warped image to the result image
        # choose the best blending by comparing to golden
        # blend for every pic 0~3
        mask1 = np.any(warping_part1 != [0, 0, 0], axis=-1)
        for k in range(0, h//16):
            for j in range(0, w//16):
                true_count = np.sum(mask1[k*16:(k+1)*16, j*16:(j+1)*16])
                false_count = 256 - true_count
                if (true_count >= false_count):
                    mask1[k*16:(k+1)*16, j*16:(j+1)*16] = True
                else:
                    mask1[k*16:(k+1)*16, j*16:(j+1)*16] = False
        for k in range(0, h//16):
            for j in range(0, w//16):
                if (a[k, j] != 0):
                    mask1[k*16:(k+1)*16, j*16:(j+1)*16] = False

        result_image1[mask1] = warping_part1[mask1]
        mask2 = np.any(warping_part2 != [0, 0, 0], axis=-1)  # (2160, 3840)
        for k in range(0, h//16):
            for j in range(0, w//16):
                true_count = np.sum(mask2[k*16:(k+1)*16, j*16:(j+1)*16])
                false_count = 256 - true_count
                if (true_count >= false_count):
                    mask2[k*16:(k+1)*16, j*16:(j+1)*16] = True
                else:
                    mask2[k*16:(k+1)*16, j*16:(j+1)*16] = False
        for k in range(0, h//16):
            for j in range(0, w//16):
                if (a[k, j] != 0):
                    mask2[k*16:(k+1)*16, j*16:(j+1)*16] = False

        result_image2[mask2] = warping_part2[mask2]
        # do blending for the same objects by choosing blending coefficient
        alpha_values = np.linspace(0, 10, num=11)  # num can be modified
        MSE = 0
        best_alpha = 0
        min_MSE = float('inf')
        for alpha in alpha_values:
            alpha = alpha/10
            blended = alpha*result_image1 + (1-alpha)*result_image2
            diff = blended-golden
            MSE = np.sum(diff**2)
            if MSE < min_MSE:
                best_alpha = alpha
                min_MSE = MSE
                blended_image = blended
                # update blended_image
                # need to save the value of alpha
        blendings_list.append(best_alpha)
        cnt += 1

        # check occupy
        for k in range(0, h//16):
            for j in range(0, w//16):
                if (np.any(blended_image[k*16:(k+1)*16, j*16:(j+1)*16] != 0) and (a[k, j] == 0)):
                    a[k, j] = cnt
        result_image1 = blended_image
        result_image2 = blended_image

    # after the 4th pic
    Hs1 = Hs1[4:]
    Hs2 = Hs2[4:]
    masks_r_dir1 = masks_r_dir1[4:]
    masks_r_dir2 = masks_r_dir2[4:]
    model_arrange = []
    for j in range(len(Hs1)):
        warping_part1 = warping_single(
            image_r_dir1, Hs1[j], masks_r_dir1[j], j)
        golden_image_part = np.where(warping_part1 != 0, golden, 0)
        error = (np.mean((golden_image_part - warping_part1) ** 2))
        model_arrange.append((error, 'before', Hs1[j], masks_r_dir1[j]))
    for j in range(len(Hs2)):
        warping_part2 = warping_single(
            image_r_dir2, Hs2[j], masks_r_dir2[j], j)
        golden_image_part = np.where(warping_part2 != 0, golden, 0)
        error = (np.mean((golden_image_part - warping_part2) ** 2))
        model_arrange.append((error, 'after', Hs2[j], masks_r_dir2[j]))
    model_arrange.sort(key=lambda x: x[0])

    for j in tqdm.tqdm(range(len(model_arrange))):
        cnt += 1
        if (cnt > 12):
            break
        error, indicater, H, mask_r_dir = model_arrange[j]
        if indicater == 'before':
            warping_part = warping_single(image_r_dir1, H, mask_r_dir, j)
            H1_list.append(H)
            H2_list.append(H)
            masks_r_dir1_list.append(mask_r_dir)
            masks_r_dir2_list.append(mask_r_dir)
            blendings_list.append(1.0)
        else:
            warping_part = warping_single(image_r_dir2, H, mask_r_dir, j)
            H1_list.append(H)
            H2_list.append(H)
            masks_r_dir1_list.append(mask_r_dir)
            masks_r_dir2_list.append(mask_r_dir)
            blendings_list.append(0.0)

        mask = np.any(warping_part != [0, 0, 0], axis=-1)

        for k in range(0, h//16):
            for j in range(0, w//16):
                true_count = np.sum(mask[k*16:(k+1)*16, j*16:(j+1)*16])
                false_count = 256 - true_count
                if (true_count >= false_count):
                    mask[k*16:(k+1)*16, j*16:(j+1)*16] = True
                else:
                    mask[k*16:(k+1)*16, j*16:(j+1)*16] = False

        blended_image[mask] = warping_part[mask]
        # check occupy
        for k in range(0, h//16):
            for j in range(0, w//16):
                if mask[k*16:(k+1)*16, j*16:(j+1)*16].all() == True:
                    a[k, j] = cnt
    blended_image = blended_image.astype(np.uint8)

    def save_data(homographies1, homographies2, masks_r1_paths, masks_r2_paths, blendings, filename):
        data = {
            'homographies1': homographies1,
            'masks_r1_paths': masks_r1_paths,
            'homographies2': homographies2,
            'masks_r2_paths': masks_r2_paths,
            'blendings': blendings
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    # store the homography matrix and corresponding mask_dir in the same json file to reduce the time of calculation
    model_map_path = f'./model_map/m_{dir_t}.txt'
    data_file = f'./model_dir/{dir_t}.pkl'
    save_data(H1_list, H2_list, masks_r_dir1_list,
              masks_r_dir2_list, blendings_list, data_file)

    model_map = []
    for k in range(0, h//16):
        for j in range(0, w//16):
            model_map.append(int(a[k, j]))
    model_map_str = list(map(str, model_map))
    with open(model_map_path, 'w') as f:
        f.write("\n".join(model_map_str))
    plt.imsave(f'./prediction/{dir_t}.png', blended_image, cmap='gray')


# Main
dir_t = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062',
         '063', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '097', '098', '099', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127']

dir_r1 = ['000', '000', '002', '000', '004', '004', '006', '000', '008', '008', '010', '008', '012', '012', '014', '000', '016', '016', '018', '016', '020', '020', '022', '016', '024', '024', '026', '024', '028', '028', '030', '032', '032', '034', '032', '036', '036', '038', '032', '040', '040', '042', '040', '044', '044', '046', '032', '048', '048', '050', '048', '052', '052', '054', '048', '056', '056', '058', '056', '060', '060',
          '062', '064', '064', '066', '064', '068', '068', '070', '064', '072', '072', '074', '072', '076', '076', '078', '064', '080', '080', '082', '080', '084', '084', '086', '080', '088', '088', '090', '088', '092', '092', '094', '096', '096', '098', '096', '100', '100', '102', '096', '104', '104', '106', '104', '108', '108', '110', '096', '112', '112', '114', '112', '116', '116', '118', '112', '120', '120', '122', '120', '124', '124', '126']

dir_r2 = ['002', '004', '004', '008', '006', '008', '008', '016', '010', '012', '012', '016', '014', '016', '016', '032', '018', '020', '020', '024', '022', '024', '024', '032', '026', '028', '028', '032', '030', '032', '032', '034', '036', '036', '040', '038', '040', '040', '048', '042', '044', '044', '048', '046', '048', '048', '064', '050', '052', '052', '056', '054', '056', '056', '064', '058', '060', '060', '064', '062', '064',
          '064', '066', '068', '068', '072', '070', '072', '072', '080', '074', '076', '076', '080', '078', '080', '080', '096', '082', '084', '084', '088', '086', '088', '088', '096', '090', '092', '092', '096', '094', '096', '096', '098', '100', '100', '104', '102', '104', '104', '112', '106', '108', '108', '112', '110', '112', '112', '128', '114', '116', '116', '120', '118', '120', '120', '128', '122', '124', '124', '128', '126', '128', '128']

total_psnr = []
for i in range(len(dir_t)):
    print(f'Excecuting {dir_t[i]}...')
    execute(dir_t[i], dir_r1[i], dir_r2[i])
