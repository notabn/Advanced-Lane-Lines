import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import collections
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


dist_pickle = pickle.load(open('dist_pickle.p','rb'))

mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]



def abs_sobel_thresh(channel, orient='x', sobel_size=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0,ksize=sobel_size)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1,ksize=sobel_size)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
    return grad_binary

# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(channel, sobel_size=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=sobel_size)
    abs_sobelxy = np.sqrt(sobel_x**2 + sobel_y ** 2)
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    return binary_output


def dir_threshold(chanel, sobel_size=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    sobel_x = cv2.Sobel(chanel, cv2.CV_64F, 1, 0, ksize=sobel_size)
    sobel_y = cv2.Sobel(chanel, cv2.CV_64F, 0, 1, ksize=sobel_size)
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary

def calc_x_offset(img,offset_x,lane_width):
    xm_per_pix = 3.7/lane_width
    center_img_x = img.shape[0]
    center_lane = offset_x + lane_width/2
    offset_pix = center_img_x-center_lane
    offset_m = offset_pix * xm_per_pix
    return offset_m

# Edit this function to create your own pipeline.
def pipeline(img, s_thresh=(160, 230),l_thresh = (200, 210), sx_thresh=(35, 255),ksize=15,plot = False):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    R = undist[:, :, 0]
    thresh_red = (221, 255)
    binary_red = np.zeros_like(R)
    binary_red[(R >= thresh_red[0]) & (R <= thresh_red[1])] = 1

    # Convert to HLV color space and separate the V channel
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Threshold gradient
    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_size = ksize, thresh=sx_thresh)
    mag_binary = mag_thresh(l_channel, sobel_size=ksize, mag_thresh=(60, 100))
    dir_binary = dir_threshold(l_channel, sobel_size=ksize, thresh=(0.7, 1.3))

    grad_binary = np.zeros_like(dir_binary)
    grad_binary[((gradx == 1)  | ((mag_binary == 1) & (dir_binary == 1)))] = 1


    # Threshold color channel

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel < s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( binary_red, s_binary,grad_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[ (binary_red == 1) | (grad_binary == 1)| (s_binary == 1) ] = 1

    if plot:

        plt.figure()
        plt.title('Combined S,R channel and gradient thresholds')
        plt.imshow(combined_binary, cmap='gray')
        plt.imsave('./images/binary_combo_example')

        f, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(20, 10))
        ax1.set_title('Original image')
        ax1.imshow(img)

        ax2.set_title('Combined S,R channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')

        ax3.set_title('Combined S channel and gradient thresholds')
        ax3.imshow(color_binary, cmap='gray')

    # Perspective Transform
    warp_img, M, Minv = warper(combined_binary)

    if plot:
        #output = np.array(cv2.merge((warped, warped, warped)))
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Original image')
        ax1.imshow(img)
        ax2.set_title('Warped image')
        ax2.imshow(warp_img)

    warp_binary = cv2.cvtColor(warp_img, cv2.COLOR_RGB2GRAY)


    update_search = True
    count = 0
    veh_offset = 0
    left_fit= []
    right_fit = []
    if len(left_lane.current_fit)>0 and len(right_lane.current_fit)>0 :
        left_fit, right_fit = find_following_lines(warp_binary, left_lane.current_fit, right_lane.current_fit)
    else:
        left_fit, right_fit = find_first_lines(warp_binary)



    while update_search:
        if len(left_fit ) > 0 and len(right_fit) >0 :
            ploty = np.linspace(0, warp_binary.shape[0] - 1, warp_binary.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

            y_eval = np.max(ploty)
            lane_left_px_pos = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
            lane_right_px_pos = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
            lane_width = lane_right_px_pos - lane_left_px_pos
            veh_offset = round(calc_x_offset(undist,lane_left_px_pos,lane_width),2)
            #print('veh offset {}'.format(veh_offset))

            # Define conversions in x and y from pixels space to meters
            ym_per_pix = 30 / 720  # meters per pixel in y dimension
            xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

            # Fit new polynomials to x,y in world space
            left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
            right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
            # Calculate the new radii of curvature
            left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * left_fit_cr[0])
            right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
                2 * right_fit_cr[0])
            # Now our radius of curvature is in meters
            #print(left_curverad, 'm', right_curverad, 'm')

            check = True
            eps = 2500
            # check for similar curvature
            if abs(left_curverad - right_curverad) > eps:
                check = check & False
            # check distance between lanes
            lane_width = 800
            current_lane_width = np.mean(abs(left_fitx- right_fitx))

            if current_lane_width > lane_width :
                check = check & False

            if check:
                left_lane.update_current_fit(left_fit,left_curverad,warp_binary)
                right_lane.update_current_fit(right_fit,right_curverad,warp_binary)
                update_search = False

            if not check and (len(left_lane.current_fit) > 0 and left_lane.detected == False):
                left_lane.recover_last_fit(warp_binary)
                right_lane.recover_last_fit(warp_binary)
                update_search = False

        if update_search:
            left_fit, right_fit = find_following_lines(warp_binary, left_fit, right_fit)
        if count > 5:
            break
        count +=1

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.allx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.allx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #Output visual display of the lane  boundaries and numerical estimation of lane curvature and vehicle position.
    text = 'Radius of curvature ' +str(round(left_lane.radius_of_curvature,2)) +' m'
    cv2.putText(result,text,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255))
    text = 'Vehicle is '+str(veh_offset)+ ' m left of center'
    cv2.putText(result,text,(100,150),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255))
    if plot :
        plt.figure()
        plt.imshow(result)
    return result


def warper(img, plot = False):

    img_size = (img.shape[1],img.shape[0])
    #print(img.shape)

    # Manually chosen source points along the lane marking
    top_left_src = [590, 450]
    top_right_src  = [686, 450]
    bottom_right_src  = [1132, 720]
    bottom_left_src  = [200, 720]

    #top_left_src = [590, 450]
    #top_right_src  = [680, 450]
    #bottom_right_src  = [1100, 720]
    #bottom_left_src  = [230, 720]

    # destination points
    top_left_dst = [320, 10]
    top_right_dst = [980, 10]
    bottom_right_dst = [980, 720]
    bottom_left_dst = [320, 720]

    src = np.array([bottom_left_src , bottom_right_src , top_right_src , top_left_src ])
    dst = np.array([bottom_left_dst, bottom_right_dst , top_right_dst , top_left_dst ],np.float32)

    # Draw the windows on the visualization image
    img = np.array(img,np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #cv2.polylines(img, [src], True,(255,0,0), 1)

    src = np.float32(src)
    # Perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Inverse perspective transform
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    # Plotting images
    if plot :
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Original image')
        #output_img = np.array(cv2.merge((img, img, img)))
        ax1.imshow(img)
        ax2.set_title('Warped image')
        #output = np.array(cv2.merge((warped, warped, warped)))
        ax2.imshow(warped)


    return warped, M, Minv

def find_first_lines(binary_warped,plot=False):

    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype('uint8')* 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 12
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []


    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if plot :
        plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit,right_fit

def find_following_lines(binary_warped,left_fit,right_fit,plot=False):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 50
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)).astype('uint8')* 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    if plot:
        plt.figure()
        plt.title('window search')
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    return left_fit,right_fit


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):

    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
            :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids

def sliding_window_search(warped,plot=False):

    # window settings
    window_width = 80
    window_height = 80  # Break image into 12 vertical layers since image height is 720
    margin = 40  # How much to slide left and right for searching

    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    left_fitx = []
    right_fitx = []

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage =  np.array(cv2.merge((warped, warped, warped)), np.uint8)* 255 # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

        left_lane_ind = np.where(l_points==255)
        right_lane_ind = np.where(r_points==255)

        leftx = left_lane_ind[1]
        lefty = left_lane_ind[0]
        rightx = right_lane_ind[1]
        righty = right_lane_ind[0]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        if plot:
            plt.figure()
            plt.title('convolution search and fitting results')
            plt.imshow(output)
            plt.plot(left_fitx, ploty, color='red')
            plt.plot(right_fitx, ploty, color='red')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)


    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)
        # Display the final results
        if plot:
            plt.figure()
            plt.imshow(output)
            plt.title('window fitting results')
            plt.show()
    return left_fit, right_fit


# Define a class to receive the characteristics of each line detection
class Line():
    n = 9
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = collections.deque(maxlen=10)
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([])
        # polynomial coefficients of the last n fits of the line
        self.recent_fit = collections.deque(maxlen=10)
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

    def update_current_fit(self,coefficients, radius,img):
        if len(self.current_fit) > 0:
            self.diffs = self.current_fit - coefficients
        self.detected = True
        self.current_fit = coefficients
        self.radius_of_curvature = radius
        self.recent_fit.append(coefficients)
        self._fit_data(img)
        self.smoothing()



    def calculate_radius(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit( self.ally * ym_per_pix, self.allx * xm_per_pix, 2)
        y_eval = np.max(self.ally)
        self.radius_of_curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * fit_cr[0])


    def _fit_data(self,img):
        self.ally = np.linspace(0, img.shape[0] - 1, img.shape[0])
        self.allx = self.current_fit[0] * self.ally ** 2 + self.current_fit[1] * self.ally + self.current_fit[2]
        for x in self.allx[-10:]:
            self.recent_xfitted.append(x)



    def recover_last_fit(self,img):
        self.detected = False
        self.current_fit = self.best_fit
        self.recent_xfitted.append(self.bestx)
        self._fit_data(img)
        self.calculate_radius()
        self.smoothing()
        self.diffs = 0


    def smoothing(self):
        self.bestx = np.mean(self.recent_xfitted)
        a = 0
        b = 0
        c = 0
        for idx,coef in enumerate(self.recent_fit):
            a += coef[0]
            b += coef[1]
            c += coef[2]
        idx += 1
        self.best_fit = np.array([a/idx,b/idx,c/idx])


images = glob.glob('test_images/test1.jpg')

for fname in images:
    img = cv2.imread(fname)

    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
    #ax1.imshow(img)
    #ax1.set_title('Original Image', fontsize=30)
    #ax2.imshow(undist)
    #ax2.set_title('Undistorted Image', fontsize=30)
    left_lane = Line()
    right_lane = Line()
    pipeline(img,plot=True)


plt.show()
print('done')
'''
left_lane = Line()
right_lane = Line()

detection_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline)  # NOTE: this function expects color images!!

white_clip.write_videofile(detection_output, audio=False)


'''
