## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./output_images/distortion1.png "Camera Calibration"
[image2]: ./output_images/undistorted.png "Undistorted Image"
[image3]: ./output_images/color_mask.jpg "Color Mask"
[image4]: ./output_images/hls_threshold.jpg "HLS Space"
[image5]: ./output_images/sobel_on_l_s_channel.jpg "Sobel Filter"
[image6]: ./output_images/combine_filter_and_smoothing.jpg "Combine Filter"
[image7]: ./output_images/perspective_transform.jpg "Perspective Transform"
[image8]: ./output_images/histogram.jpg "Histogram"
[image9]: ./output_images/lane_fitting.jpg "Polynomial Fitting"
[image10]: ./output_images/project_back.jpg "Lane Detection"
[video1]: ./project_video_out_complete.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the coefficients obtained through camera calibration, I applied distortion correction to each image. Below is one example: 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code of the process creating a thresholded binary image is in cell 8,9,10,11 of the jupyter notebook.

I first tried using yellow color and white color masks on HSV color space since most road lane are of yellow and white color. HSV color space is used because it is less sensitive to different lighting conditions. Result is as below: 

![alt text][image3]

Then I tried thresholding in HLS color space. Based on the result, L and S channel are promising in detecting lanes. 
![alt text][image4]

Therefore, I applied Sobel filter on L and S channel. Result is as below: 
![alt text][image5]

Then, I combined all those filters to get a good result. A smoothing filter is also applied to remove the noises brought by Sobel filter. Final result is shown as below:
![alt text][image6]



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform in the 7th cell of the notebook.  The `calculate_perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The function to identify the lane-line pixels and polynomial fitting are `find_line_fit_and_car_offset()` and `get_fit_xy()`.

First, I used a histogram of intensities of first row of the binary image to find the base points of lanes. The base point should be the point where the intensity is highest. Result of the histogram is as below:
![alt text][image8]

Then I used 9 windows in the vertical direction to find the lane-line pixels in y direction starting from the base points. The location of lane-line pixels are found within the window, where value is 1 in the binary thresholded image. The location of window is updated for each row based on the central location of the lane-line pixels of previous row. After these pixels are identified, a polynomial is fitted using numpy `polyfit()` function. An example is shown below: 
![alt text][image9]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated in function `calculate_curvature_and_vehicle_position()` using the mathematical equation of polynomial. The position of the vehicle is calculated using difference between the center of the image and the center of the left and right lane-line pixels. The pixel value is converted to meter using meter per pixel. 


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The function to rever the perspective transfrom and plotting the lane zone is `project_back()`. 
An example of my result on a test image is as below:

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out_complete.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I did not implement a lane tracker which would reject false detection based on previos frames. There are some minor mistakes when the car gets bumpy and lighting conditions change. This can be improved using a class to track the lane changes as suggested in the project.   
