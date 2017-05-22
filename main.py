from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import views as view
import thresholds
import glob
import matplotlib.pyplot as plt
import windows as win


class Lane(object):

    def __init__(self, image, laneline):
        self.h = image.shape[0]
        self.w = image.shape[1]
        self.lanepoints = np.zeros_like(image)
        self.lanepoints[(laneline == 255) & (image == 1)] = 1
        self.x = np.where(self.lanepoints == 1)[1]
        self.y = np.where(self.lanepoints == 1)[0]
        self.fits = None

    def fit(self):
        self.fits = np.polyfit(self.y, self.x, deg=2)
        return self.fits

    def generate_y(self):
        return np.linspace(0, self.h - 1, self.h)

    def get_points(self):
        y = self.generate_y()
        A = self.fits[0]
        B = self.fits[1]
        C = self.fits[2]
        return np.stack((
            A * y ** 2 + B * y + C,
            y
        )).astype(np.int).T

    def evaluate(self):
        y = self.generate_y()
        A = self.fits[0]
        B = self.fits[1]
        C = self.fits[2]
        return A * y ** 2 + B * y + C,

    def curvature_radius(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        y = self.get_points()[:, 1] * ym_per_pix
        x = self.get_points()[:, 0] * xm_per_pix
        y_max = 720 * ym_per_pix
        params = np.polyfit(y, x, 2)
        A = params[0]
        B = params[1]

        return int(
            ((1 + (2 * A * y_max + B)**2 )**1.5) /
            np.absolute(2 * A)
        )

    def camera_distance(self):
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        y = self.get_points()
        x = y[np.max(y[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)


def setup():
    # calibrate the camera - do NOT need to run everytime
    # camera matrix and distribution coefficients are pickled
    view.calibratecamera()

    # "undistort" calibration (chessboard) images
    view.undistort('./camera_cal/', singlefile="False")


#prepare source and destination points for the test images
h = 720
w = 1280

srcpts = np.float32([
    [0, h],
    [w / 2 - 76, h * .625],
    [w / 2 + 75, h * .625],
    [w, h]
])

dstpts = np.float32([
    [100, h],
    [100, 0],
    [w - 100, 0],
    [w - 100, h]
])

leftlines = []
rightlines = []
curvatures = []

def processframes(image):
    #files = glob.glob('./test_images/*.jpg')
    #for index, image in enumerate([cv2.imread(filename) for filename in files]):

    undistort = view.undistort(image)
    threshold = thresholds.thresholdimage(undistort)
    topview = view.convertview(threshold, srcpts, dstpts)

    windows, leftline, rightline = win.sliding_window(topview)

    #window_centroids = win.find_window_centroids(topview)
    #topview = win.draw_windows(topview, window_centroids)

    leftlane = Lane(topview, leftline)
    rightlane = Lane(topview, rightline)

    leftlane.fit()
    rightlane.fit()

    pace = 8

    leftlines.append(leftlane)
    rightlines.append(rightlane)

    no_threshold = view.convertview(undistort, srcpts, dstpts)

    y = leftlane.generate_y()

    drawing = np.zeros_like(no_threshold)
    lx = np.median(np.array([l.evaluate() for l in leftlines[-pace:]]), axis=0)
    rx = np.median(np.array([l.evaluate() for l in rightlines[-pace:]]), axis=0)

    left_points = np.vstack([lx, y]).T
    right_points = np.vstack([rx, y]).T

    all_points = np.concatenate([left_points, right_points[::-1], left_points[:1]])

    cv2.fillConvexPoly(drawing, np.int32([all_points]), (0, 255, 0))

    no_warp_drawing = view.convertview(drawing, srcpts, dstpts, inverse="True")

    frame = cv2.addWeighted(undistort, 1.0, no_warp_drawing, 0.2, 0)

    l = np.average(np.array([line.camera_distance() for line in leftlines[-pace:]]))
    r = np.average(np.array([line.camera_distance() for line in rightlines[-pace:]]))
    if l - r > 0:
        cv2.putText(frame, '{:.3} cm right of center'.format((l - r) * 100), (20, 115), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, '{:.3} cm left of center'.format((r - l) * 100), (20, 115), cv2.FONT_HERSHEY_SIMPLEX, .8,(255, 255, 255), 2)

    curvatures.append(np.mean([leftlane.curvature_radius(), rightlane.curvature_radius()]))
    curvature = np.average(curvatures[-pace:])
    cv2.putText(frame, 'Radius of curvature:  {:.3} km'.format(curvature / 1000), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, .8,(255, 255, 255), 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

    '''
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax2.imshow(frame)

    ax1.set_title('Original')
    ax2.set_title('Thresholds applied')
    plt.show()

    f.savefig('test {:02}'.format(index) + '.jpg')
    '''


# Load the video and feed frames to the pipeline
video = VideoFileClip('challenge_video.mp4')
processed_video = video.fl_image(processframes)
processed_video.write_videofile("output.mp4", audio=False)
