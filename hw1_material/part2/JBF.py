import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # spatial kernel gs
        xq = np.arange(0, self.wndw_size)
        xq = xq[:, np.newaxis] # self.wndw_size*1 array
        yq = np.arange(0, self.wndw_size)
        yq = yq[np.newaxis, :] # 1*self.wndw_size array
        gs = np.exp(-((xq-self.pad_w)**2 + (yq-self.pad_w)**2)/(2*(self.sigma_s**2)))

        # pixel values normalized to [0,1]
        padded_guidance = padded_guidance/255

        output = np.zeros_like(img) # output same size as img
        h = guidance.shape[0] # = img.shape[0]
        w = guidance.shape[1] # = img.shape[1]

        for x in range(self.pad_w, h+self.pad_w):
            for y in range(self.pad_w, w+self.pad_w): # x(row), y(column) is the center of a window
                window = padded_img[x-self.pad_w:x+self.pad_w+1, y-self.pad_w:y+self.pad_w+1] # self.wndw_size*self.wndw_size*3 array
                # range kernel gr
                if(guidance.ndim == 3): # 3 channels(rgb) case
                    f1 = padded_guidance[x, y]
                    f2 = padded_guidance[x-self.pad_w:x+self.pad_w+1, y-self.pad_w:y+self.pad_w+1]
                    # gr = np.exp(np.sum(-((f2-f1)**2)/(2*(self.sigma_r**2)), axis = 2))
                    gr = np.exp((-((f2-f1)**2)/(2*(self.sigma_r**2))).sum(axis = 2))
                else: # 1 channel(gray) case
                    f1 = padded_guidance[x, y]
                    f2 = padded_guidance[x-self.pad_w:x+self.pad_w+1, y-self.pad_w:y+self.pad_w+1]
                    gr = np.exp((-(f2-f1)**2)/(2*(self.sigma_r**2)))
                
                tmp = gs*gr
                W = tmp.sum()
                               
                # output (faster)
                output[x-self.pad_w, y-self.pad_w, 0] = (tmp*window[:,:,0]).sum() / W
                output[x-self.pad_w, y-self.pad_w, 1] = (tmp*window[:,:,1]).sum() / W
                output[x-self.pad_w, y-self.pad_w, 2] = (tmp*window[:,:,2]).sum() / W
                # output (slower)
                # tmp = tmp[:, :, np.newaxis] # become self.wndw_size*self.wndw_size*1 array
                # output[x-self.pad_w, y-self.pad_w, :] = (tmp*window).sum(axis=(0, 1)) / W

        return np.clip(output, 0, 255).astype(np.uint8)
