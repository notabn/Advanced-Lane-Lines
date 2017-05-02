
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

[image1]: ./images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./images/binary_combo_example.png "Binary Example"
[image4]: ./images/warped_straight_lines.png "Warp Example"
[image5]: ./images/color_fit_lines.jpg "Fit Visual"
[image6]: ./images/example_output.png "Output"
[video1]: ./project_video_output.mp4.mp4 "Video"


### README


### Camera Calibration

The code for this step is contained in the in lines 12 through 58 of the file called `calibrate_camera.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2.Generate thresholded binary image. 

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 73 through 105 in `lane_detection.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective transform 

The code for my perspective transform includes a function called `warper()`, which appears in lines 237 through 308 in the file `lane_dection.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.array(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 50, img_size[1]],
         [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
    
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 203.33, 720      | 320, 720      |
| 1116.66, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4.Identified lane-line pixels and fit their positions with a polynomial

To identify the lane-lines pixes I first take a histogram along all the columns in the lower half of the image and the peaks of the histogramm  will be good indicators of the x-position of the base of the lane lines.
Then with a sliding window I identify the 'hot' pixels, with the initial position of the window at the left and right peaks of the histogramm. The pixels insid this window are then identified as
lane-line pixels. Further I fitted a second order polynomial to the identified pixels as in the following image depicted as a yellow line 

![alt text][image5]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature of the lane is calculated in lines​​ 167 through 172 in `lane_detection.py`

I calculated  the position of the vehicle with respect to the center in my code in in `lane_detection.py` in the function `calc_x_offset()` 

#### 6. Detected area.

I implemented this step in lines 224 through 234 in my code in `lane_detection.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

I first identified the lane-line pixels as described in point 4. In the pipeline the next search is in a margin around the previous line position coded in function `find_following_lines()`. Also a check to verify if the width of the lines and if
the lines are parellel is done. If the check is not pass another search will be performed. If also in this case the check fails a fallback to the previous check is done and the coefficients mean over the last 9 values are taken.
The implementation doesn't perform at its best when there is no contrast between the lines and the road. However with the smothing the algorithm is more robust to different lightness conditions when the lanes are not shown in the
the generate thresholded binary image.