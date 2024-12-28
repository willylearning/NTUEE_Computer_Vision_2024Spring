import os
import cv2
import numpy as np

def detect_and_draw_features(image, mask):
    # 確保遮罩是二值的
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # 初始化SIFT
    sift = cv2.SIFT_create()

    # 使用遮罩檢測特徵點和描述符
    keypoints, descriptors = sift.detectAndCompute(image, mask)

    # 繪製特徵點
    output_image = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 創建一個全黑的圖像
    black_background = np.zeros_like(image)

    # 將遮罩內的區域從原始圖像複製到黑色背景上
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    black_background[mask == 255] = masked_image[mask == 255]

    # 在黑色背景上繪製特徵點
    output_image = cv2.drawKeypoints(black_background, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return output_image

def process_directory(image_path, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍歷資料夾中的所有遮罩圖像
    for mask_filename in os.listdir(input_dir):
        if mask_filename.endswith('.png'):
            mask_path = os.path.join(input_dir, mask_filename)
            # image_path = '000.png' # 這裡是原始圖像的路徑

            # 讀取原始圖像和遮罩圖像
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Failed to load image or mask at path: {image_path} or {mask_path}")
                continue

            # 調整遮罩尺寸以匹配原始圖像尺寸
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 檢測特徵點並黑化背景
            output_image = detect_and_draw_features(image, mask)

            # 保存結果圖像
            output_path = os.path.join(output_dir, mask_filename)
            cv2.imwrite(output_path, output_image)
            print(f"Processed and saved: {output_path}")

# 路徑到你的資料夾
image_path = '128.png'
input_dir = '.\seg_generate\seg_generate_128'
output_dir = '.\seg_generate_withfeature\seg_generate_128_withfeature'

# 處理資料夾
process_directory(image_path, input_dir, output_dir)
