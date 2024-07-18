import numpy as np
import cv2
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    # image padding
    padded_Il = cv2.copyMakeBorder(Il, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0) # (h+2, w+2, ch)
    padded_Ir = cv2.copyMakeBorder(Ir, 1, 1, 1, 1, borderType=cv2.BORDER_CONSTANT, value=0)

    # image patch -> local binary pattern
    Il_code = np.zeros((h, w, 8, ch), dtype=np.bool) # type should be bool in order to use ^ (xor)
    Ir_code = np.zeros((h, w, 8, ch), dtype=np.bool)
    for i in range(1, h+1):
        for j in range(1, w+1): # clockwise order (start from the left point)
            Il_code[i-1, j-1, 0] = padded_Il[i-1, j-1] < padded_Il[i, j]
            Il_code[i-1, j-1, 1] = padded_Il[i-1, j] < padded_Il[i, j]
            Il_code[i-1, j-1, 2] = padded_Il[i-1, j+1] < padded_Il[i, j]
            Il_code[i-1, j-1, 3] = padded_Il[i, j+1] < padded_Il[i, j]          
            Il_code[i-1, j-1, 4] = padded_Il[i+1, j+1] < padded_Il[i, j]
            Il_code[i-1, j-1, 5] = padded_Il[i+1, j] < padded_Il[i, j]
            Il_code[i-1, j-1, 6] = padded_Il[i+1, j-1] < padded_Il[i, j]
            Il_code[i-1, j-1, 7] = padded_Il[i, j-1] < padded_Il[i, j]

            Ir_code[i-1, j-1, 0] = padded_Ir[i-1, j-1] < padded_Ir[i, j]
            Ir_code[i-1, j-1, 1] = padded_Ir[i-1, j] < padded_Ir[i, j]
            Ir_code[i-1, j-1, 2] = padded_Ir[i-1, j+1] < padded_Ir[i, j]
            Ir_code[i-1, j-1, 3] = padded_Ir[i, j+1] < padded_Ir[i, j]          
            Ir_code[i-1, j-1, 4] = padded_Ir[i+1, j+1] < padded_Ir[i, j]
            Ir_code[i-1, j-1, 5] = padded_Ir[i+1, j] < padded_Ir[i, j]
            Ir_code[i-1, j-1, 6] = padded_Ir[i+1, j-1] < padded_Ir[i, j]
            Ir_code[i-1, j-1, 7] = padded_Ir[i, j-1] < padded_Ir[i, j]

    # Census cost = Local binary pattern -> Hamming distance
    l_cost = np.zeros((h, w, max_disp+1), dtype=np.float32)
    r_cost = np.zeros((h, w, max_disp+1), dtype=np.float32)

    for d in range(max_disp + 1):
        for i in range(h):
            for j in range(w):
                if(j-d >= 0):
                    l_cost[i, j, d] = np.sum((Il_code[i, j]^Ir_code[i, j-d]).astype(np.uint8), axis=(0, 1)) # need .astype(np.uint8) to transform bool to int
                else: # cannot go left d
                    l_cost[i, j, d] = l_cost[i, d, d]
    for d in range(max_disp + 1):
        for i in range(h):
            for j in range(w):
                if(j+d <= w-1):
                    r_cost[i, j, d] = np.sum((Ir_code[i, j]^Il_code[i, j+d]).astype(np.uint8), axis=(0, 1))
                else: # cannot go right d
                    r_cost[i, j, d] = r_cost[i, w-1-d, d]

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    for d in range(max_disp+1):
        l_cost[:, :, d] = xip.jointBilateralFilter(Il, l_cost[:, :, d], -1, 4, 12)
        r_cost[:, :, d] = xip.jointBilateralFilter(Ir, r_cost[:, :, d], -1, 4, 12)  

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    D_L = np.argmin(l_cost, axis=2) # (h, w)
    D_R = np.argmin(r_cost, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    # Left-right consistency check
    # Note: D_R are only used in this step
    for y in range(h):
        for x in range(w): 
            if (x-D_L[y, x] >= 0) and (D_L[y, x] == D_R[y, x-D_L[y,x]]):
                continue   # keep the computed disparity
            else: 
                D_L[y, x] = -1  # mark hole (invalid disparity)

    # Hole filling
    # pad 1 row each at leftmost and rightmost
    padded_D_L = cv2.copyMakeBorder(D_L, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=max_disp) # (h, w+2)
    for y in range(h):
        for x in range(w):
            if D_L[y, x] == -1:
                l = 1                
                while(padded_D_L[y, x+1-l] == -1): # padded_D_L[y, x+1-l] = D_L[y, x-l]
                    l += 1
                F_L = padded_D_L[y, x+1-l] # the disparity map filled by closest valid disparity from left

                r = 1
                while(padded_D_L[y, x+1+r] == -1): # padded_D_L[y, x+1+r] = D_L[y, x+r]
                    r += 1
                F_R = padded_D_L[y, x+1+r] # the disparity map filled by closest valid disparity from right

                D_L[y, x] = min(F_L, F_R) # pixel-wise minimum

    # Weighted median filtering
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), D_L.astype(np.uint8), 18, 3)

    return labels.astype(np.uint8)