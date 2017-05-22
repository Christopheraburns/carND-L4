"""
The views.py file will handle:
    loading calibration data
    camera calibration
    "undistorting" an image
    perspective transform of images
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import os
import pickle

# Empty arrays to hold img and Obj points of calibration images
imgpoints = []
objpoints = []

objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)


def loadcalibrationdata():
    # Check for the camera.p file
    if os.path.isfile('camera.p'):
        with open('camera.p', 'rb') as pickle_file:
            data = pickle.load(pickle_file)
        return data['mtx'], data['dist']


def loadcalibrationimages(path, savetodisk="False"):
    # Array to return to calling function with corners marked
    imgCorners = []
    nx = 9
    ny = 6
    images = os.listdir(path)
    for fname in images:
        img = cv2.imread(path + "\\" + fname)

        # convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners
            i = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

            # Add the images with corners to the image array
            imgCorners.append(i)

            # Save images to disk
            if savetodisk == "True":
                # save the undistored image
                # append "corners" to filename
                cv2.imwrite(path + "\\" + "corners_" + fname, i)
        else:
            print("Cannot find corners for: {}".format(fname))
    return imgCorners


def calibratecamera(display='False', savetodisk='False'):
    imgUndistort = []
    print('loading calibration images')
    calibrationImages = loadcalibrationimages('./camera_cal/')
    print('calibration images loaded... calibrating camera...')
    x = 1
    for img in calibrationImages:
        # Convert to Grayscale for calibration
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        imgUndistort.append(undist)
        if savetodisk == "True":
            cv2.imwrite('dist_corrected' + str(x) + '.jpg', undist)
        x += 1

    # calibrate camera with a test image
    cc_img = mpimg.imread('./test_images/straight_lines1.jpg')
    cc_size = (cc_img.shape[1], cc_img.shape[2])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cc_size, None, None)
    # Store mtx and dist
    print('camera calibrated... storing matrix')
    cc_pickle = {}
    cc_pickle["mtx"] = mtx
    cc_pickle["dist"] = dist
    pickle._dump(cc_pickle, open("camera.p", "wb"))

    if display == "True":
        # Show the first 4 images before and after distortion correction
        plt.subplot(241), plt.title("Image1 - distorted"), plt.imshow(calibrationImages[0])
        plt.subplot(242), plt.title("Image2 - distorted"), plt.imshow(calibrationImages[1])
        plt.subplot(243), plt.title("Image3 - distorted"), plt.imshow(calibrationImages[2])
        plt.subplot(244), plt.title("Image4 - distorted"), plt.imshow(calibrationImages[3])
        plt.subplot(245), plt.title("Image1 - Undistorted"), plt.imshow(imgUndistort[0])
        plt.subplot(246), plt.title("Image2 - Undistorted"), plt.imshow(imgUndistort[1])
        plt.subplot(247), plt.title("Image3 - Undistorted"), plt.imshow(imgUndistort[2])
        plt.subplot(248), plt.title("Image4 - Undistorted"), plt.imshow(imgUndistort[3])
        plt.show()


def convertview(img, srcpts, dstpts, singlefile="True", displayresults="False", inverse="False"):
    #print("converting images to top-down view")
    converted = []
    if singlefile == "True":
        try:
            #print('applying undistort function')
            # "Target" should already be an image - assuming BGR format
            image_size = (img.shape[1], img.shape[0])
            M = cv2.getPerspectiveTransform(srcpts, dstpts)
            warped = cv2.warpPerspective(img, M, image_size)

            if inverse == "True":
                M = cv2.getPerspectiveTransform(dstpts, srcpts)
                inverse = cv2.warpPerspective(img, M, image_size)
                return inverse
            else:
                return warped
        except Exception as err:
            print("Error converting to top down view!: {}".format(err))
    else:
        for image in img:
            image_size = (image.shape[1], image.shape[0])
            M = cv2.getPerspectiveTransform(srcpts, dstpts)
            warped = cv2.warpPerspective(image, M, image_size)
            converted.append(warped)
    return converted


def undistort(target, singlefile='True', displayresults='False',
              savetodisk='False', findcorners='False', perspectivetransform='False'):

    if singlefile == "True":  # undistort single file and return results
        try:
            #print('applying undistort function')
            # "Target" should already be an image - assuming BGR format
            img = np.copy(target)
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
            if findcorners == "True":
                # Add chessboard corners overlay
                # convert to grayscale
                gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
                # find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                if ret:
                    # Draw and display the corners
                    undist = cv2.drawChessboardCorners(undist, (9, 6), corners, ret)
                    # can only do a perspective transform with the following code IF corners have been found
                    if perspectivetransform == "True":
                        offset = 100
                        img_size = (gray.shape[1], gray.shape[0])

                        srcpts = np.float32([
                            corners[0],
                            corners[8],
                            corners[-1],
                            corners[-9]
                        ])
                        dstpts = np.float32([
                            [offset, offset],
                            [img_size[0] - offset, offset],
                            [img_size[0] - offset, img_size[1] - offset],
                            [offset, img_size[1] - offset]
                        ])
                        M = cv2.getPerspectiveTransform(srcpts, dstpts)
                        undist = cv2.warpPerspective(undist, M, img_size)
            if displayresults == "True":
                # Show before and after image
                plt.subplot(121), plt.title('Original'), plt.imshow(img)
                plt.subplot(122), plt.title('Undistorted'),plt.imshow(undist)
                plt.show()
            if savetodisk == "True":
                # save the undistored image
                # append undist to filename
                cv2.imwrite("undistorded.jpg", undist)
        except Exception as err:
            print("Error while applying undistort to an image!: {}".format(err))
        return undist
    else:
        undistorted = []
        #print('applying undistort function to all files in directory: ' + target)
        # Target is a path, undistort all images
        try:
            images = os.listdir(target)
            for fname in images:
                img = cv2.imread(target + fname)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                undist = cv2.undistort(img, mtx, dist, None, mtx)
                if findcorners == "True":
                    # Add chessboard corners overlay
                    # convert to grayscale
                    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
                    # find the chessboard corners
                    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                    if ret:
                        # Draw and display the corners
                        undist = cv2.drawChessboardCorners(undist, (9, 6), corners, ret)
                        # can only do a perspective transform with the following code IF corners have been found
                        if perspectivetransform == "True":
                            offset = 100
                            img_size = (gray.shape[1], gray.shape[0])

                            srcpts = np.float32([
                                corners[0],
                                corners[8],
                                corners[-1],
                                corners[-9]
                            ])
                            dstpts = np.float32([
                                [offset, offset],
                                [img_size[0] - offset, offset],
                                [img_size[0] - offset, img_size[1] - offset],
                                [offset, img_size[1] - offset]
                            ])
                            M = cv2.getPerspectiveTransform(srcpts, dstpts)
                            undist = cv2.warpPerspective(undist, M, img_size)
                if displayresults == "True":
                    # Show before and after image
                    plt.subplot(121), plt.title('Original'), plt.imshow(img)
                    plt.subplot(122), plt.title('Undistorted'), plt.imshow(undist)
                    plt.show()
                if savetodisk == "True":
                    # save the undistored image
                    # append undist to filename
                    name = "undist_"
                    if findcorners == "True":
                        name = "corners_" + name
                    if perspectivetransform == "True":
                        name = "warped_" + name
                    cv2.imwrite(target + name + fname, undist)
                undistorted.append(undist)
        except Exception as err:
            print("Error while applying undistort to a directory of images!: {}".format(err))
        return undistorted


mtx, dist = loadcalibrationdata()