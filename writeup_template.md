## Writeup 

---

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

[image1]: ./output_images/distortion1.png "Undistorted1"
[image2]: ./output_images/distortion2.png "Undistorted2"
[image3]: ./output_images/undistortion_straight_lane.png "Road Transform"
[image4]: ./output_images/combining_filters.png "Combining Filters"
[image5]: ./output_images/perspective_transform.png "Perspective Transform"
[image6]: ./output_images/perspective_transform_curved.png "Warp Example"
[image7]: ./output_images/find_lane_plot.png "Fit Visual"
[image8]: ./output_images/pipeline_on_another_test.png "Output"

[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"




## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./findlane.ipynb" 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  I used glob module to ensemble all the points.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
After we get ret, mtx, dist, rvecs, tvecs from camera calibration, we can apply the undistortion. 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (abs_sobel_thresh, mag_threshold, dir_threshold, and hls_select defined in the 7th cell of notebook).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `apply_perspective_transform()` (or in the 8th cell as an example before the function is defined later).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([[600,450],[680,450], [1080, img_size[1]], [200, img_size[1]]])
dst = np.float32([[350,0],[1000,0],[1000, img_size[1]],[350, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 600, 450      | 350, 0        | 
| 680, 450      | 1000, 0      |
| 1080, 720     | 1000, 720      |
| 200, 460      | 350, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I followed instructions of the project to find the lane in the test images: 
1. first apply undistortion and threshold (`undistort(img, mtx, dist)` and `apply_threshold(img, ksize, thresh, mag_thresh, sobel_kernel, angle_thesh, hls_thresh)` defined in the notebook)
2. get the warped binary image by applying perspective_transform
![alt text][image6]
3. get the start point of lane at the bottom of the image and move upwards by finding the lane point within a window, 
and finally fit my lane lines with a 2nd order polynomial kinda like this:
![alt text][image7]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I defined a function `calculate_curvature` to do this, which basically is the helper function provided in the lecture. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_zone()`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out_complete.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
The current implementation assume the camera lens location is fixed. If the lens moved while driving, the camera needs to be recalibrated and images would need another set of coefficients to undistort. 
