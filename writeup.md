

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corners_calibration2.jpg "DrawChessBoardCorners"
[image2]: ./output_images/un-distorted19%20(13).jpg "Road Transformed"
[image3]: ./output_images/00.jpg ""
[image4]: ./output_images/topview00%20(1).jpg ""
[image5]: ./output_images/slidingwindows%2007%20(2).jpg ""
[image6]: ./output_images/lanepaint%20(1).jpg "Painted lane"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to calibrate the camera is contained in the views.py file.  All the associated steps are encapsulated into two functions.  calibratecamera() and loadcalibrationimages().  I utilize the 20 images of chess boards to 
perform the calibration. I first run the loadcalibrationimges() function. This function contains the following key line of code:
```python
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```
gray is the image converted to grayscale.  NX and NY are the number of interior corners (9 and 6).  I save the corners value into an imgpoints array for later use
 
 I iterate through the 20 images and execute the following two lines of code on each image: 
```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undist = cv2.undistort(img, mtx, dist, None, mtx)
```
The following image is an example of the cv2.drawChessboardCorners method used after the cv2.findChessboardCorners. This image has not yet received distortion correction

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
To perform this distortion correction i created a function called "undistort" the function signature is:
```python
def undistort(target, singlefile='True', displayresults='False', savetodisk='False', findcorners='False', perspectivetransform='False'):
```
The method can accept a single image for processing, or a directory of images.
All the "undistorting" happens in a single line of code:
```python
undist = cv2.undistort(img, mtx, dist, None, mtx)
```
where "img" is the image to modify and "mtx" and "dist" are the camera matrix and distribution coefficients that we stored in the calibratecamera() function


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I have a dedicated .py file for applying color transforms and gradients called thresholds.py.  This file has several functions that perform a specific task or apply a specific transformation to the image:
```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=20, thresh_max=100):
```
This applies the cv2.Sobel function to the image on either the x or y axis
```python
def mag_thresh(img, sobel_kernel=3, mag_thresh=(20,100)):
```
This applies a Magnitude threshold to both x and y axis of a Sobel

```python
def dir_threshold(img, sobel_kernel=3, mag_thresh=(0.7, 1.3)):
```
This applies a directional threshold with this code:
```python
absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
```

A final method within the thresholds.py file orchestrates the calling of these three functions and also combines the results into a single image

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called:
```python
def convertview(img, srcpts, dstpts, singlefile="True", displayresults="False", inverse="False"):
```
The two key lines of code in this function are:
```python
M = cv2.getPerspectiveTransform(srcpts, dstpts)
warped = cv2.warpPerspective(img, M, image_size)
```
where M is the matrix resulting from passing the source and destination points.
`warped` is the value that holds the altered image.

I do not hard code the required source and destination points 

```python
h = image.shape[0]
w = image.shape[1]

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
```
The points are created based on the image H and W and passed to the convertview function.


I verified that my perspective transform was working as expected applying binary thresholds to the image below and then applying the perspective transform to clearly see the lane lines from a topdown view


![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding windows method that we learned in the lesson to identify the lane lines, once I applied a binary threshold and a perspective transformation
I dedicated a third file to fitting polynomials. It is called windows.py

Here is the function I used to apply the sliding windows to my images:

```python
def sliding_window(image, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, margin=MARGIN):
```



![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I wrote this code last and left it in the main.py file - which serves as the orchestration point of my pipeline.  Here is the function I used to calculate curvature:

```python
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

```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
For the final application of the lane coloring, I created a Lane class within the main.py file.  On line #138 of main.py I apply the lane coloring:

```python
cv2.fillConvexPoly(drawing, np.int32([all_points]), (0, 255, 0))
```

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's an online [link to my video result](https://www.youtube.com/watch?v=HQIrTX80EO8)

Here's a [local copy](./advanced_lane_detection.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
I had a *very* difficult time applying the lane coloring (fitting polynomials) to the image. The last 10% of the project (working up to the cv2.fillConvexPoly call) took 90% of the time.  This pipeline really only works with this video
  as the lane lines in the videos require additional work in the thresholding and gradient stages.
