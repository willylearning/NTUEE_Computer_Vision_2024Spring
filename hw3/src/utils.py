import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2*N, 9))
    # deal with odd rows
    A[0::2, 0:2] = u[0::1, 0:2]
    A[0::2, 2] = 1
    A[0::2, 6:8] = -(u[0::1, 0:2])*(v[0::1, 0:1])
    A[0::2, 8] = -(v[0::1, 0])
    # deal with even rows
    A[1::2, 3:5] = u[0::1, 0:2]
    A[1::2, 5] = 1
    A[1::2, 6:8] = -(u[0::1, 0:2])*(v[0::1, 1:2])
    A[1::2, 8] = -(v[0::1, 1])

    # TODO: 2.solve H with A
    U, S, VT = np.linalg.svd(A)
    H = VT[-1, :].reshape((3, 3)) # Let H be the last column of V <=> last row of VT
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xx, yy = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    x = xx.reshape(((xmax-xmin)*(ymax-ymin), 1))
    y = yy.reshape(((xmax-xmin)*(ymax-ymin), 1))
    ones = np.ones(((xmax-xmin)*(ymax-ymin), 1))
    U = np.hstack((x, y, ones)) # N x 3

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H_inv, U.T) # mapping in src (change to 3 x N)
        V = V/V[2]  
        Vx = V[0].reshape((ymax-ymin, xmax-xmin))
        Vy = V[1].reshape((ymax-ymin, xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        # mask = v (logical operation) image range
        mask = ((Vx >= 0) & (Vx <= w_src-1)) & ((Vy >= 0) & (Vy <= h_src-1)) 

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # output = do something * mask
        outputx = Vx[mask] # shape: (n, )
        outputy = Vy[mask]

        # Bilinear interpolation
        i = np.floor(outputx).astype(int)
        j = np.floor(outputy).astype(int)
        a = (outputx-i)[:, np.newaxis] # shape: (n, 1)
        b = (outputy-j)[:, np.newaxis]
        output = np.zeros((h_src, w_src, ch)) # same shape as src
        output[j, i] += (1-a)*(1-b)*src[j, i] + a*(1-b)*src[j, i] + \
                            a*b*src[j+1, i+1] + (1-a)*b*src[j, i+1]

        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax,xmin:xmax][mask] = output[j, i]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H, U.T) # mapping in dst (change to 3 x N)
        V = np.round(V/V[2]).astype(int) # may be non-integer, so need to do round. Also, np.round() returns float64
        Vx = V[0].reshape((ymax-ymin, xmax-xmin))
        Vy = V[1].reshape((ymax-ymin, xmax-xmin))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        # mask = v (logical operation) image range
        mask = ((Vx >= 0) & (Vx <= w_dst-1)) & ((Vy >= 0) & (Vy <= h_dst-1)) 

        # TODO: 5.filter the valid coordinates using previous obtained mask
        # output = do something * mask
        outputx = Vx[mask] 
        outputy = Vy[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[outputy, outputx] = src[mask]
        # print(dst.shape) # (1275, 1920, 3)
        # print(dst[outputy, outputx].shape) # (563000, 3)
        # print(outputy.shape) # (563000,)
     
    return dst 
