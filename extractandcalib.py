import numpy as np
import cv2 as cv
import os
import glob
import sys
import re
import shutil
from pathlib import Path 
import math

'''
This script is used to undistort a video using a checkerboard calibration. This based on the OpenCV tutorial:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
The script can be used in 3 different ways:
1. Calibrate the camera using a checkerboard pattern
2. Undistort a video using the calibration
3. Create a video from a folder of frames
'''

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def calibrate(matrix_x, matrix_y, image_dir, output_folder):
    global globmtx
    global globdist
    globmtx = None
    globdist = None
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    matrix_x = int(matrix_x)
    matrix_y = int(matrix_y)
    objp = np.zeros((matrix_x * matrix_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:matrix_x, 0:matrix_y].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob(image_dir)
    if len(images) == 0:
        print("No images found!")
        exit()
    for fname in images:
        print("Processing image: " + fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (matrix_x, matrix_y), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
            globmtx = mtx
            globdist = dist
        else:
            print("Couldn't find checkerboard!")
    if globmtx is None:
        print("Couldn't perform checkerboard camera calibration!")
        print("Please check your images and try again.")
    else:
        print("Saving calibration...")
        np.savetxt(output_folder + "mtx.txt", globmtx)
        np.savetxt(output_folder + "dist.txt", globdist)
        print("Camera Calibration Done!")


def videoprocessing(video_path, output_folder, name, crop=False):
    print("Processing video...")
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    i = 0
    frame_path = os.path.join(output_folder, "frames")
    try:
        shutil.rmtree(frame_path, ignore_errors=True)
    except:
        pass
    os.mkdir(frame_path)
    print("Calibrating...")
    try: 
        globdist
        globmtx
    except:
        dist_path = os.path.join(output_folder, "dist.txt")
        globdist = np.loadtxt(dist_path)
        mtx_path = os.path.join(output_folder, "mtx.txt")
        globmtx = np.loadtxt(mtx_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            #Find new camera matrix and undistort
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(globmtx, globdist, (w, h), 1, (w, h))
            dst = cv.undistort(frame, globmtx, globdist, None, newcameramtx)
            # crop the image
            if crop:
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]
            cv.imwrite(frame_path + "/frame" + str(i) + ".jpg", dst)
            i += 1
            print("Frame " + str(i) + " processed!")
            
        else:
            break
    cap.release()
    cv.destroyAllWindows()
    print("Video Processing Done!")
    frames2video(output_folder, fps, name)

def frames2video(path, fps, name):
    print("Creating video...")
    img_array = []
    size = None
    frame_path = os.path.join(path, "frames/*.jpg")
    for filename in sorted(glob.glob(frame_path), key=get_order):   
        print("Processing frame: " + filename )    
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    path = os.path.join(path, name)
    out = cv.VideoWriter(path + '.mp4', cv.VideoWriter_fourcc(*'avc1'), int(fps), size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Video Created!")

if sys.argv[1] == "--calibrate":
    CHECKERBOARD_FOLDER = sys.argv[2]
    CHECKERBOARD_SIZE_X = sys.argv[3]
    CHECKERBOARD_SIZE_Y = sys.argv[4]
    OUTPUT_FOLDER = sys.argv[5]
    calibrate(CHECKERBOARD_SIZE_X, CHECKERBOARD_SIZE_Y, CHECKERBOARD_FOLDER, OUTPUT_FOLDER)
elif sys.argv[1] == "--videoprocessing":
    FILE_INPUT = sys.argv[2]
    OUTPUT_FOLDER = sys.argv[3]
    NAME = sys.argv[4]
    CROP = sys.argv[5]
    videoprocessing(FILE_INPUT, OUTPUT_FOLDER, NAME, CROP)
elif sys.argv[1] == "--frames2video":
    OUTPUT_FOLDER = sys.argv[2]
    FPS = sys.argv[3]
    NAME = sys.argv[4]
    frames2video(OUTPUT_FOLDER, FPS, NAME)
elif sys.argv[1] == "--all":
    CHECKERBOARD_FOLDER = sys.argv[2]
    CHECKERBOARD_SIZE_X = sys.argv[3]
    CHECKERBOARD_SIZE_Y = sys.argv[4]
    OUTPUT_FOLDER = sys.argv[5]
    FILE_INPUT = sys.argv[6]
    NAME = sys.argv[7]
    CROP = sys.argv[8]
    calibrate(CHECKERBOARD_SIZE_X, CHECKERBOARD_SIZE_Y, CHECKERBOARD_FOLDER, OUTPUT_FOLDER)
    videoprocessing(FILE_INPUT, OUTPUT_FOLDER, NAME, CROP)
else:
    print("Command not found. Here are a list of commands:")
    print("--all CHECKERBOARD_FOLDER CHECKERBOARD_SIZE_X CHECKERBOARD_SIZE_Y OUTPUT_FOLDER VIDEO_INPUT:     Calibrate the camera, process a video, and create a video from a folder of frames")
    print("--calibrate CHECKERBOARD_FOLDER_PATH CHECKERBOARD_SIZE_X CHECKERBOARD_SIZE_Y OUTPUT_FOLDER:     Calibrate the camera and produce an intrinsic matrix and distortion coefficients")
    print("--videoprocessing VIDEO_PATH OUTPUT_FOLDER NAME:     Process a video with an existing intrinsic matrix and distortion coefficients")
    print("--frames2video FRAMES_PATH FPS NAME:     Create a video from a folder of frames")
    exit()



#calibrate(CHECKERBOARD_SIZE_X, CHECKERBOARD_SIZE_Y, CHECKERBOARD_FOLDER)
#videoprocessing(FILE_INPUT, OUTPUT_FOLDER)
#frames2video(OUTPUT_FOLDER)

    
