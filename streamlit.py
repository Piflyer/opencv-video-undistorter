import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import io
import os
import streamlit_ext as ste

st.write("""
# Video Calibration
### Undo the effects of camera distortion on videos. See this project on [GitHub](https://github.com/Piflyer/opencv-video-undistorter).
""")
st.session_state.runcount = False
tab1, tab2, tab3 = st.tabs(["Calibrate Camera", "Calibrate + Process Video", "Process Video with Matrices"])
def calibrate(matrix_x, matrix_y, calibration_img):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    objp = np.zeros((matrix_x * matrix_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:matrix_x, 0:matrix_y].T.reshape(-1, 2)
    for calib_img in calibration_img:
        global runcount
        runcount = True
        st.write("Processing image: " + calib_img.name)
        image = Image.open(calib_img)
        image = image.save("image.jpg")
        calib_img = cv.imread("image.jpg")
        gray = cv.cvtColor(calib_img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (matrix_x, matrix_y), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            ret, mtx, dist, _, _ = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
            globmtx = mtx
            globdist = dist
            st.write(globmtx)
    os.remove("image.jpg")
    if globmtx is not None:
        st.write("### Calibration successful!")
        st.write("You can now download the intrinsic matrix and distortion coefficients.")
        with io.BytesIO() as mtx_file:
            np.savetxt(mtx_file, globmtx)
            mtx_file.seek(0)
            #st.download_button("Download Intrinsic Matrix", mtx_file, file_name="mtx.txt")
            ste.download_button("Download Intrinsic Matrix", mtx_file, file_name="mtx.txt")
        with io.BytesIO() as dist_file:
            np.savetxt(dist_file, globdist)
            dist_file.seek(0)
            #st.download_button("Download Distortion Coefficients", dist_file, file_name="dist.txt")
            ste.download_button("Download Distortion Coefficients", dist_file, file_name="dist.txt")
with tab1:
    st.write("""
    ## Calibrate Camera
    ### Calibrate the camera and produce files for intrinsic matrix and distortion coefficients
    """)
    with st.container():
        st.write("""
        #### Step 1: Set the number of squares in the checkerboard
        Set the number of square to calibrate the camera. It should be the number of square on each side of the checkerboard minus one.
        """)
        matrix_x = st.number_input("Number of squares in the x direction", min_value=1, max_value=100)
        matrix_y = st.number_input("Number of squares in the y direction", min_value=1, max_value=100)
    with st.container():
        st.write("""
        #### Step 2: Upload images of a checkerboard
        Upload images of a checkerboard. The checkerboard should be fully seen in each image. The images should be taken from different angles and distances from the camera. 10-12 images should be enough.
        """)
        calibration_img = st.file_uploader("Upload images of a checkerboard", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="calibrate_button")
        st.button("Calibrate Camera", on_click=calibrate, args=(matrix_x, matrix_y, calibration_img), key="button")
    
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objpoints = []
        # imgpoints = []
        # objp = np.zeros((matrix_x * matrix_y, 3), np.float32)
        # objp[:, :2] = np.mgrid[0:matrix_x, 0:matrix_y].T.reshape(-1, 2)
        # if calibration_img is not None:
        #     for calib_img in calibration_img:
        #         image = Image.open(calib_img)
        #         image = image.save("image.jpg")
        #         st.write("Processing image: " + calib_img.name)
        #         calib_img = cv.imread("image.jpg")
        #         gray = cv.cvtColor(calib_img, cv.COLOR_BGR2GRAY)
        #         ret, corners = cv.findChessboardCorners(gray, (matrix_x, matrix_y), None)
        #         if ret == True:
        #             objpoints.append(objp)
        #             corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #             imgpoints.append(corners2)
        #             ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
        #             globmtx = mtx
        #             globdist = dist
        #         os.remove("image.jpg")
        #     if globmtx is None:
        #         st.write("Couldn't perform checkerboard camera calibration!")
# if 'submit' in st.session_state:
#     with st.container():
#         if globmtx is None:
#             st.write("### Calibration failed!")
#             st.write("Please try again with different images. Make sure the checkerboard is fully visible in each image and that the images are taken from different angles and distances from the camera.")
#         else:
#             st.write("### Calibration successful!")
#             st.write("You can now download the intrinsic matrix and distortion coefficients.")
#             with io.BytesIO() as mtx_file:
#                 np.savetxt(mtx_file, globmtx)
#                 mtx_file.seek(0)
#                 st.download_button("Download Intrinsic Matrix", mtx_file, file_name="mtx.txt")
#             with io.BytesIO() as dist_file:
#                 np.savetxt(dist_file, globdist)
#                 dist_file.seek(0)
#                 st.download_button("Download Distortion Coefficients", dist_file, file_name="dist.txt")
            

           
                
        

with tab2:
    st.write("""
    ## Calibrate + Process Video
    ### Calibrate the camera and process a video
    """)
    st.write("Coming soon...")
    # with st.container():
    #     st.write("""
    #     #### Step 1: Set the number of squares in the checkerboard
    #     Set the number of square to calibrate the camera. It should be the number of square on each side of the checkerboard minus one.
    #     """)
    #     matrix_x = st.number_input("Number of squares in the x direction", min_value=1, max_value=100, key="matrix_x")
    #     matrix_y = st.number_input("Number of squares in the y direction", min_value=1, max_value=100, key="matrix_y")
    # with st.container():
    #     st.write("""
    #     #### Step 2: Upload images of a checkerboard
    #     Upload images of a checkerboard. The checkerboard should be fully seen in each image. The images should be taken from different angles and distances from the camera. 10-12 images should be enough. Step 3 will be shown after the calibration is done.
    #     """)
        # calibration_img = st.file_uploader("Upload images of a checkerboard", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="calibration_img")
        # globmtx = None
        # globdist = None
        # criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # objpoints = []
        # imgpoints = []
        # objp = np.zeros((matrix_x * matrix_y, 3), np.float32)
        # objp[:, :2] = np.mgrid[0:matrix_x, 0:matrix_y].T.reshape(-1, 2)
        # if calibration_img is not None:
        #     for calib_img in calibration_img:
        #         image = Image.open(calib_img)
        #         image = image.save(calib_img.name)
        #         st.write("Processing image: " + calib_img.name)
        #         calib_img = cv.imread(calib_img.name)
        #         gray = cv.cvtColor(calib_img, cv.COLOR_BGR2GRAY)
        #         ret, corners = cv.findChessboardCorners(gray, (matrix_x, matrix_y), None)
        #         if ret == True:
        #             objpoints.append(objp)
        #             corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #             imgpoints.append(corners2)
        #             ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)    
        #             globmtx = mtx
        #             globdist = dist
        #     if globmtx is None:
        #         st.write("Couldn't perform checkerboard camera calibration! Please try again with different images.")
        #     else:
        #         with st.container():
        #             st.write("""
        #             #### Step 3: Upload a video
        #             # Upload a video to process. The video should be taken with the same camera as the images used for calibration.""")
        #             video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video")
                        


with tab3:
    st.write("""
    ## Process Video with Matrices
    ### Process a video with an existing intrinsic matrix and distortion coefficients""")
    st.write("Coming soon...")
