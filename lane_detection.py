# Code for lane detection 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from abc import ABCMeta

#TODO: PUT ALL CONSTANTS TO A CLASS
class Line(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        ret, mtx, dist, rvecs, tvecs = self.calibrate_camera()
        M, Minv = self.calculate_perspective_transform()
        self._ret = ret
        self._mtx = mtx
        self._dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs
        self._M = M
        self._Minv = Minv

    @property
    def ret(self):
        return self._ret

    @property
    def mtx(self):
        return self._mtx

    @property
    def dist(self):
        return self._dist

    @property
    def rvecs(self):
        return self._rvecs

    @property
    def tvecs(self):
        return self._tvecs

    @property
    def M(self):
        return self._M

    @property
    def Minv(self):
        return self._Minv

    def calibrate_camera(self):
        #read in all images
        images = glob.glob('camera_cal/calibration*.jpg')
        objpoints = []
        imgpoints = []
        objp = np.zeros((6*9,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        for fname in images:
            #read in each image
            img = mpimg.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

    def undistort(self,img):
        undistorted_image = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undistorted_image

    def calculate_perspective_transform(self, img):
        """apply perspective transform"""
        src = np.float32([[600,450],[680,450], [1080, img.shape[0]], [200, img.shape[0]]])
        src = src.reshape((-1,1,2))
        dst = np.float32([[350,0],[1000,0],[1000, img.shape[0]],[350, img.shape[0]]])
        M = cv2.getPerspectiveTransform(src,dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv

    def apply_perspective_transform(self, img):
        img_size = (img.shape[1], img.shape[0])
        img_warped = cv2.warpPerspective(img, self.M, img_size, flags=cv2.INTER_LINEAR)
        return img_warped

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        else:
            raise Exception('no such orientation')
        abs_sobel = np.absolute(sobel)

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    def mag_threshold(self, img, sobel_kernel=3, mag_thresh=(0, 255)):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobelmag = np.sqrt(np.sum([np.square(sobelx), np.square(sobely)], axis=0))
        scaled_sobel = np.uint8(255 * sobelmag / np.max(sobelmag))
        mag_binary = np.zeros_like(scaled_sobel)
        mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        return mag_binary

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi / 2)):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        sobel_angle = np.arctan2(abs_sobely, abs_sobelx)
        dir_binary = np.zeros_like(sobel_angle)
        dir_binary[(sobel_angle >= thresh[0]) & (sobel_angle <= thresh[1])] = 1
        return dir_binary

    def r_select(self, img, thresh=(200, 255)):
        R = img[:, :, 0]
        binary = np.zeros_like(R)
        binary[(R > thresh[0]) & (R <= thresh[1])] = 1
        return binary

    def color_mask(self,hsv, low, high):
        """Return color mask from HSV image"""
        mask = cv2.inRange(hsv, low, high)
        return mask

    def apply_color_mask(self, hsv, img, low, high):
        """Apply color mask to HSV image"""
        mask = cv2.inRange(hsv, low, high)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

    def apply_yellow_white_mask(self, img, yellow_hsv_low=np.array([0, 80, 200]), yellow_hsv_high=np.array([40, 255, 255]),
                                white_hsv_low=np.array([20, 0, 200]), white_hsv_high=np.array([255, 80, 255])):
        """Apply yellow white mask to HSV image"""
        image_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask_yellow = color_mask(image_HSV, yellow_hsv_low, yellow_hsv_high)
        mask_white = color_mask(image_HSV, white_hsv_low, white_hsv_high)
        mask_YW_image = cv2.bitwise_or(mask_yellow, mask_white)
        return mask_YW_image

    def hls_select(self, img, channel='S', thresh=(0, 255)):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if channel == 'S':
            channel_img = hls[:, :, 2]
        elif channel == 'H':
            channel_img = hls[:, :, 0]
        elif channel == 'L':
            channel_img = hls[:, :, 1]
        else:
            raise Exception('Ilegal channel')
        binary_output = np.zeros_like(channel_img)
        binary_output[(channel_img > thresh[0]) & (channel_img <= thresh[1])] = 1
        return binary_output

    def apply_sobel_filter_on_hls(self, img, channel='S', sobel_kernel=3, mag_thresh=(0, 255)):
        """apply sobel filter on L and S channel"""
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        if channel == 'S':
            channel_img = hls[:, :, 2]
        elif channel == 'L':
            channel_img = hls[:, :, 1]
        else:
            raise Exception('Ilegal channel')
        channel_mag_binary = mag_threshold(channel_img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh)
        return channel_mag_binary

    def combine_filters(self, wrap_img):
        """combine filter to detect lane"""
        sobel_on_s_binary = self.apply_sobel_filter_on_hls(wrap_img, mag_thresh=(15, 255))
        sobel_on_l_binary = self.apply_sobel_filter_on_hls(wrap_img, channel='L', mag_thresh=(15, 255))
        l_binary = self.hls_select(wrap_img, channel='L', thresh=(100, 200))
        s_binary = self.hls_select(wrap_img, channel='S', thresh=(100, 255))
        yw_binary = self.apply_yellow_white_mask(wrap_img)
        yw_binary[(yw_binary != 0)] = 1
        combined_lsx = np.zeros_like(sobel_on_s_binary)
        combined_lsx[(
        (l_binary == 1) & (s_binary == 1) | (sobel_on_s_binary == 1) | (sobel_on_l_binary == 1) | (yw_binary == 1))] = 1
        return combined_lsx

    def apply_smoothing(self, img, window_size=30, threshold=0.05):
        """apply smoothing in x direction and apply threshold"""
        kernel = np.ones((1, window_size), np.float32) / window_size
        dst = cv2.filter2D(img, -1, kernel)
        img_lane = np.zeros_like(dst)
        img_lane[dst >= threshold] = 1
        return img_lane

    def find_line_fit_and_car_offset(self, img, nwindows=9, margin=100, minpix=50):
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        plt.plot(histogram)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((img, img, img)) * 255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        offset = 1 / 2 * (leftx_base + rightx_base) - midpoint
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # to plot
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        return left_fit, right_fit, out_img, offset

    def get_fit_xy(self, img, left_fit, right_fit):
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        return left_fitx, right_fitx, ploty

    def project_back(self, wrap_img, origin_img, left_fitx, right_fitx, ploty, Minv):
        warp_zero = np.zeros_like(wrap_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = apply_perspective_transform(color_warp, Minv)
        # Combine the result with the original image
        result = cv2.addWeighted(origin_img, 1, newwarp, 0.3, 0)
        return result

    def calculate_curvature_and_vehicle_position(self, ploty, left_fitx, right_fitx, offset_pixel):
        y_eval = np.max(ploty)
        # pixel to real world
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = (
                         (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        offset = offset_pixel * xm_per_pix
        return left_curverad, right_curverad, offset

    def add_curvature_and_offset_info(self, result, left_curverad, right_curverad, offset):
        cv2.putText(result, "left radius:" + str(int(left_curverad)) + 'm', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0))
        cv2.putText(result, "right radius:" + str(int(right_curverad)) + 'm', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 0))
        cv2.putText(result, "offset:" + str(round(offset, 2)) + 'm', (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
        return result

    def detect_lane_from_frame(self, frame):
        #undistort
        frame_undistored = self.undistort(frame)
        #apply perspective transform
        frame_warped = self.apply_perspective_transform(frame_undistored)
        if not self.recent_xfitted:
            #apply filter and smoothing
            lane_filtered = self.combine_filters(frame_warped)
            lane_smoothed = self.apply_smoothing(lane_filtered)
            #find left lane and right lane
            left_fit, right_fit, out_img, offset_pixel = self.find_line_fit_and_car_offset(lane_smoothed)
            left_fitx, right_fitx, ploty = self.get_fit_xy(lane_smoothed, left_fit, right_fit)
            frame_w_lane = self.project_back(lane_smoothed, frame, left_fitx, right_fitx, ploty, Minv)
            left_curverad, right_curverad, offset = self.calculate_curvature_and_vehicle_position(ploty, left_fitx, right_fitx, offset_pixel)
            frame_w_lane_and_info = self.add_curvature_and_offset_info(frame_w_lane, left_curverad, right_curverad, offset)
        else:
            frame_w_lane_and_info = self.find_line_fit_and_car_offset_based_on_previous_frame()
        return frame_w_lane_and_info

    def sanity_check(self, margin):
        pass

    def look_ahead_filter(self, margin):
        """ Search for the new line within +/- some margin around the old line center"""
        pass

    def reset(self):
        """keep the previous position from prior frame for bad or difficult frame"""
        pass

    def smoothing(self):
        """"""
        pass

    def find_line_fit_and_car_offset_based_on_previous_frame(self):
        """find lane based on previous lane"""
        pass


def process_image(img):
    lane = Line()
    #use class method to find lane
    img_with_lane = lane.detect_lane_from_frame(img)
    return img_with_lane


clip = VideoFileClip("project_video.mp4")
clip_annotated = clip.fl_image(process_image)
clip_annotated.write_videofile("project_video_out.mp4", audio=False)



