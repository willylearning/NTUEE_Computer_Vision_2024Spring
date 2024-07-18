import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
from utils import solve_homography, warping


def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    h, w, c = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # TODO: find homography per frame and apply backward warp
    pbar = tqdm(total = 353)
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            # TODO: 1.find corners with aruco
            # function call to aruco.detectMarkers()
            corners, ids, rejected = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
            # corners: A list containing the 4 (x, y)-coordinates, and elements in corners are float32 type
            corner = corners[0][0].astype(int) # 4 x 2 array, and elements in corner are int type

            # print(type(corners)) # list
            # print(type(corner)) # ndarray 
            # print(corners[0][0][0]) # a point
            # print(corner[0]) # a point

            # TODO: 2.find homograpy
            # function call to solve_homography()
            H = solve_homography(ref_corns, corner)

            # TODO: 3.apply backward warp
            # function call to warping()
            ymin, ymax = np.min(corner[:, 1]), np.max(corner[:, 1])
            xmin, xmax = np.min(corner[:, 0]), np.max(corner[:, 0])

            frame = warping(ref_image, frame, H, ymin, ymax, xmin, xmax, direction='b')

            videowriter.write(frame)
            pbar.update(1)

        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/arknights.png' 
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)