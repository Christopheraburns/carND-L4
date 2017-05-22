"""
thresholds.py will be used to process all color and gradient work
"""
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

from sobel import abs_sobel_thresh


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=20, thresh_max=100):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else: #y gradient
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Absolute value of the X derivative:
    abs_sobel = np.absolute(sobel)

    # Convert absolute value to an 8 bit image
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(20,100)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    #Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    scale_factor = np.max(gradmag)/255

    gradmag = (gradmag / scale_factor).astype(np.uint8)
    sxbinary = np.zeros_like(gradmag)
    sxbinary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return sxbinary

def dir_threshold(img, sobel_kernel=3, mag_thresh=(0.7, 1.3)):

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    #Take the absolute value of the gradient direction
    #Apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= mag_thresh[0]) & (absgraddir <= mag_thresh[1])] = 1
    return binary_output

def combinethresholds(img):
    # prepare gradients in LUV and LAB
    # Credit to Tim Camber for the LUV + LAB method instead of HLS

    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv).astype(np.float)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab).astype(np.float)

    l_channel = luv[:, :, 0]
    b_channel = lab[:, :, 2]

    lxgradient = abs_sobel_thresh(l_channel)
    lygradient = abs_sobel_thresh(l_channel, orient='y')
    lmagnitude = mag_thresh(l_channel)
    ldirectional = dir_threshold(l_channel)

    bxgradient = abs_sobel_thresh(b_channel)
    bygradient = abs_sobel_thresh(b_channel, orient='y')
    bmagnitude = mag_thresh(b_channel)
    bdirectional = dir_threshold(b_channel)

    gradient = np.zeros_like(l_channel)
    gradient[
        ((lxgradient == 1) & (lygradient == 1)) |
        ((lmagnitude == 1) & (ldirectional == 1)) |
        ((bxgradient == 1) & (bygradient == 1)) |
        ((bmagnitude == 1) & (bdirectional == 1))
        ] = 1

    # Add binary mask
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= 145) & (b_channel <= 255)] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 230) & (l_channel <= 255)] = 1

    color = np.dstack((l_binary, gradient, b_binary))

    combined = np.zeros_like(l_channel)
    combined[(l_binary >= 1) | (gradient >= 1) | (b_binary >=1)] = 1

    return combined


def thresholdimage(target, singlefile='True', displayresults='False', savetodisk='False'):
    if singlefile == "True":
        try:
            img = np.copy(target)

            combined = combinethresholds(img)

            if displayresults == "True":
                plt.subplot(121), plt.title('Original'), plt.imshow(img)
                plt.subplot(122), plt.title('Thresholds'), plt.imshow(combined)
                plt.show()

            return combined
        except Exception as err:
            print("Error while trying to apply thresholds to an image! : {}",format(err))
    else:
        thresholds = []
        try:
            files = glob.glob(target + '*.jpg')
            for index, image in enumerate([cv2.imread(filename) for filename in files]):

                combined = combinethresholds(image)

                if displayresults == "True":
                    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    ax2.imshow(combined)

                    ax1.set_title('Original')
                    ax2.set_title('Thresholds applied')
                    plt.show()

                    if savetodisk == "True":
                        f.savefig(target + '{:02}'.format(index) + '.jpg')
        except Exception as err:
            print("Error while trying to apply thresholds to images! : {}", format(err))
        return thresholds
