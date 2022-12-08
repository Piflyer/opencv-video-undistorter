# OpenCV Video Undistorter
This is a simple tool to undistort videos using OpenCV. The program first calibrates the camera using a set of images of a checkerboard to find the intrinsic matrix and distortion parameters. It breaks down each video frame by frame and undistorts each frame using the calibration parameters. The undistorted video is saved to a new file. 

This project is based on this [OpenCV Tutorial.](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)

## Installation
1) Clone the repository

``git clone ``

2) Install the requirements

``pip install -r requirements.txt``

## Usage
### Calibrate and Perform Video Undistortion
``python3 extractandcalib.py --all CB_FOLDER CB_X CB_Y OUTPUT VIDEO``

* ``CB_FOLDER``: Folder containing checkerboard images that will be used to compute the camera calibration parameters. The path must end with a wildcard ex. ``/path/to/folder/*``.
* ``CB_X``: Number of inner corners plus one in the checkerboard pattern in the x direction.
* ``CB_Y``: Number of inner corners plus one in the checkerboard pattern in the y direction.
* ``OUTPUT``: Output folder for undistorted images and video.
* ``VIDEO``: Video to be undistorted.

This will perform camera calibration, undistort the video, and save the undistorted video to a new file in the ``OUTPUT`` folder.

### Calibrate Camera
``python3 extractandcalib.py --calibrate CB_FOLDER CB_X CB_Y OUTPUT``
* ``CB_FOLDER``: Folder containing checkerboard images that will be used to compute the camera calibration parameters. The path must end with a wildcard ex. ``/path/to/folder/*``.
* ``CB_X``: Number of inner corners plus one in the checkerboard pattern in the x direction.
* ``CB_Y``: Number of inner corners plus one in the checkerboard pattern in the y direction.
* ``OUTPUT``: Output folder for intrinsic matrix and distortion parameters.

This will only perform camera calibration on the checkerboard images and save the intrinsic matrix and distortion parameters to a file in the ``OUTPUT`` folder. These parameters can then be used to undistort videos using the ``--undistort`` option.

### Undistort Video
``python3 extractandcalib.py --videoprocessing VIDEO OUTPUT``
* ``OUTPUT``: Output folder for undistorted images and video.
* ``VIDEO``: Video to be undistorted.

This will only undistort the video using the intrinsic matrix and distortion parameters in the ``OUTPUT`` folder. The video will be saved to a new file in the ``OUTPUT`` folder.

### Frames to Video
``python3 extractandcalib.py --frames2video FRAMES_PATH FPS NAME``
* ``FRAMES_PATH``: Path to folder containing undistorted frames.
* ``FPS``: Frames per second of the video.
* ``NAME``: Name of the video file.

This will convert the undistorted frames to a video and save it to a new file in the ``OUTPUT`` folder.

## License
[MIT](LICENSE)
