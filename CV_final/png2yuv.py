import os
import numpy as np
from PIL import Image
import shutil
import platform


def insert_golden_images(png_dir, golden_dir):
    golden_indices = [0, 32, 64, 96, 128]
    for index in golden_indices:
        src_file = os.path.join(golden_dir, f"{index:03d}.png")
        dst_file = os.path.join(png_dir, f"{index:03d}.png")
        if os.path.exists(src_file):
            if platform.system() == "Windows":
                shutil.copy(src_file, dst_file)
            else:
                os.system(f"cp {src_file} {dst_file}")


def png_to_yuv(seq_len, png_dir, output_yuv_file, w, h):

    with open(output_yuv_file, 'wb') as f_yuv:
        for frame_num in range(seq_len):
            # Load PNG image
            img_path = os.path.join(png_dir, f"{frame_num:03d}.png")
            img = Image.open(img_path).convert('L')
            img_data = np.array(img)

            # Write Y plane
            f_yuv.write(img_data.tobytes())

            # Write U and V planes (set to 128 for grayscale)
            uv_plane = np.full((h//2, w//2), 128, dtype=np.uint8)
            f_yuv.write(uv_plane.tobytes())
            f_yuv.write(uv_plane.tobytes())


if __name__ == "__main__":
    predict_dir = './reconstruction'
    golden_dir = './golden'
    output_yuv_file = './video.yuv'

    w, h = 3840, 2160
    seq_len = 129  # Total number of images

    # Insert golden images into the PNG sequence
    insert_golden_images(predict_dir, golden_dir)

    # Convert PNG images to YUV format
    png_to_yuv(seq_len, predict_dir, output_yuv_file, w, h)
