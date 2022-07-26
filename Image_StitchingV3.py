import cv2
import numpy as np
import sys
import glob
import imutils
import matplotlib.pyplot as plt
import time


ratio=0.7
min_match=30
sift=cv2.SIFT_create()
#self.sift=cv2.ORB_create()
smoothing_window_size=100
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

def registration(img1,img2):
    
    # Detect keypoints and compute descriptors
    #img1 = cv2.convertScaleAbs(img1)
    #img2 = cv2.convertScaleAbs(img2)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY), None)
    #matcher = cv2.BFMatcher()
    #raw_matches = matcher.knnMatch(des1, des2, k=2)
    raw_matches = flann.knnMatch(des1, des2, k=2)
    
    # Filter matches using the Lowe's ratio test
    good_points = []
    good_matches=[]
    for m1, m2 in raw_matches:
        if m1.distance < ratio * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])
    #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

        
    # Estimate homography matrix
    if len(good_points) > min_match:
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,4.0)

        with open('homography.npy', 'wb') as f:
            np.save(f, np.array(H))
            
    return H, image1_kp, image2_kp, good_matches
    
    

def create_mask(img1,img2,version, smoothing_window_size=100):
    
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2
    offset = int(smoothing_window_size / 2)
    barrier = img1.shape[1] - int(smoothing_window_size / 2)
    mask = np.zeros((height_panorama, width_panorama))
    # Create mask
    if version == 'left_image':
        mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
        mask[:, :barrier - offset] = 1
    else:
        mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
        mask[:, barrier + offset:] = 1
        
    return cv2.merge([mask, mask, mask])

    
def Convert_xy(x, y):
    global center, f

    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    
    return xt, yt
    
    
    
def projectOntoCylinder(InitialImage):
    global w, h, center, f
    h, w = InitialImage.shape[:2]
    center = [w // 2, h // 2]
    f = 1100       # 1100 field; 1000 Sun; 1500 Rainier; 1050 Helens
    
    # Creating a blank transformed image
    TransformedImage = np.zeros(InitialImage.shape, dtype=np.uint8)
    
    # Storing all coordinates of the transformed image in 2 arrays (x and y coordinates)
    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]
    
    # Finding corresponding coordinates of the transformed image in the initial image
    ii_x, ii_y = Convert_xy(ti_x, ti_y)

    # Rounding off the coordinate values to get exact pixel values (top-left corner)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    # Finding transformed image points whose corresponding 
    # initial image points lies inside the initial image
    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    # Removing all the outside points from everywhere
    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]
    
    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]

    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]

    # Bilinear interpolation
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)
    
    TransformedImage[ti_y, ti_x, :] = ( weight_tl[:, None] * InitialImage[ii_tl_y,     ii_tl_x,     :] ) + \
                                    ( weight_tr[:, None] * InitialImage[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                    ( weight_bl[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                    ( weight_br[:, None] * InitialImage[ii_tl_y + 1, ii_tl_x + 1, :] )


    # Getting x coorinate to remove black region from right and left in the transformed image
    min_x = min(ti_x)

    # Cropping out the black region from both sides (using symmetricity)
    TransformedImage = TransformedImage[:, min_x : -min_x, :]

    return TransformedImage, ti_x-min_x, ti_y

    
    
def image_rectification(img1, img2, matches, kp1, kp2):
  
    # Keep good matches: calculate distinctive image features
    #matchesMask = [[0, 0] for i in range(len(matches))]
   
    # Draw the keypoint matches between both pictures
    #draw_params = dict(matchColor=(0, 255, 0),
         #           singlePointColor=(255, 0, 0),
          #          matchesMask=matchesMask,
           #         flags=cv2.DrawMatchesFlags_DEFAULT)

    #keypoint_matches = cv2.drawMatchesKnn(
        #img1, kp1, img2, kp2, matches, None, **draw_params)
    #cv2.imshow("Keypoint matches", keypoint_matches)
    #cv2.waitKey(0)

    # ------------------------------------------------------------
    # STEREO RECTIFICATION

    # Calculate the fundamental matrix for the cameras
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    # Visualize epilines
    def drawlines(img1src, img2src, lines, pts1src, pts2src):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        h, r, c = img1src.shape
        #img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
        #img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
        img1color = img1src.copy()
        img2color = img2src.copy()
        # Edit: use the same random seed so that two images are comparable!
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
            img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
        return img1color, img2color

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(
        pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(
        pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    #img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    #plt.subplot(121), plt.imshow(cv2.cvtColor(img5, cv2.COLOR_BGR2RGB))
    #plt.subplot(122), plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    #plt.suptitle("Epilines in both images")
    #plt.show()

    # Stereo rectification (uncalibrated variant)
    h1, w1, c = img1.shape
    h2, w2, c = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1))


    # Undistort (rectify) the images and save them
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    #cv2.imwrite("rectified_1.png", img1_rectified)
    #cv2.imwrite("rectified_2.png", img2_rectified)
    
    return img1_rectified, img2_rectified

        

def blending(img1,img2):
    
    H = registration(img1,img2)
    
    with open('homography.npy', 'rb') as f:
        H = np.load(f)
        
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = create_mask(img1,img2,version='left_image')
    #cv2.imshow("mask1", mask1)
    #cv2.waitKey(0)
    
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = create_mask(img1,img2,version='right_image')
    #cv2.imshow("mask2", mask2)
    #cv2.waitKey(0)
    
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
    
    result = panorama1 + panorama2

    #cv2.imshow("Result", result)
    #cv2.waitKey(0)


    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    #cv2.imshow("Imgage Stitching", final_result)
    #cv2.waitKey(0)
    
    return final_result
    
#Removing Unwanted or Black Areas around Image
def trim(img3):

    grayimage = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(grayimage,5,255,cv2.THRESH_BINARY)      
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        crop = img3[y:y+h,x:x+w]
    return crop
    
    
def imageStitching(img1,img2,img3):
    #img1 = cv2.imread(argv1)
    #img2 = cv2.imread(argv2)
    
    
    imageStitcher = Image_Stitching()
    imageStitcher.smoothing_window_size = 150

    img1_rectified, img2_rectified = imageStitcher.image_rectification(img1, img2)
    final1 = imageStitcher.blending(img1_rectified,img2_rectified)
    final1 = cv2.convertScaleAbs(final1)
    final1 = imageStitcher.trim(final1)
    cv2.imwrite('panorama1.png', final1)
    #final1,_,_ = imageStitcher.projectOntoCylinder(final1)
    cv2.imwrite('panorama1Cyl.png', final1)
    
    
    imageStitcher.smoothing_window_size = 150
    final2 = imageStitcher.blending(img2_rectified,img3)
    final2 = cv2.convertScaleAbs(final2)
    final2 = imageStitcher.trim(final2)
    cv2.imwrite('panorama2.png', final2)
    #final2,_,_ = imageStitcher.projectOntoCylinder(final2)
    cv2.imwrite('panorama2Cyl.png', final2)
 
    imageStitcher.smoothing_window_size = 50
    final3 = imageStitcher.blending(final1, img3)
    final3 = cv2.convertScaleAbs(final3)
    final3 = imageStitcher.trim(final3)
    x = 0
    y = 15
    w = 2240
    h = 880

    #final3 = final3[y:y + h, x:x + w]
    cv2.imwrite('panorama3.png', final3)
    final3,_,_ = imageStitcher.projectOntoCylinder(final3)
    cv2.imwrite('panorama3Cyl.png', final3)
    
    
    imageStitcher.smoothing_window_size = 50
    final35 = imageStitcher.blending(img1_rectified, final2)
    final35 = cv2.convertScaleAbs(final35)
    final35 = imageStitcher.trim(final35)
    final35,_,_ = imageStitcher.projectOntoCylinder(final35)
    cv2.imwrite('panorama35.png', final35)
    
    
    imageStitcher.smoothing_window_size = 50
    final4 = imageStitcher.blending(final1, final2)
    final4 = cv2.convertScaleAbs(final4)
    final4 = imageStitcher.trim(final4)
    cv2.imwrite('panorama4.png', final4)
    

    cv2.imwrite("stitchedOutputProcessed.png", final3)
    #cv2.imshow("Stitched Image Processed", final3)

    #cv2.waitKey(0)
    
    return final35


    
    
    
#image_paths = sorted(glob.glob('ImagesStitching/*.png'))
#images = []

#for image in image_paths:
   # img = cv2.imread(image)
   # images.append(img)

#img1 = images[0]
#img2 = images[1]
#img3 = images[2]

#img1 = cv2.imread("C:/Users/nhoei/knightec/sceneImages/first/imageFirst5.png")
#img2 = cv2.imread("C:/Users/nhoei/knightec/sceneImages/second/imageSecond5.png")
#img3 = cv2.imread("C:/Users/nhoei/knightec/sceneImages/third/imageThird5.png")

#img1 = cv2.imread("C:/Users/nhoei/knightec/camerasCloser/first/imageFirst0.png")
#img2 = cv2.imread("C:/Users/nhoei/knightec/camerasCloser/second/imageSecond0.png")
#img3 = cv2.imread("C:/Users/nhoei/knightec/camerasCloser/third/imageThird0.png")

img1 = cv2.imread("C:/Users/nhoei/knightec/newImages/first/imageFirst1.png")
img2 = cv2.imread("C:/Users/nhoei/knightec/newImages/second/imageSecond1.png")
img3 = cv2.imread("C:/Users/nhoei/knightec/newImages/third/imageThird1.png")

#img1 = cv2.imread("C:/Users/nhoei/knightec/rectified_1.png")
#img2 = cv2.imread("C:/Users/nhoei/knightec/rectified_2.png")

#img1 = cv2.imread("panorama1.jpg")
#img2 = cv2.imread("panorama2.jpg")


stitched_image = imageStitching(img1,img2,img3)
   

"""cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(3)
cap3 = cv2.VideoCapture(2)

num = 0

while cap1.isOpened():

    succes1, img1 = cap1.read()
    succes2, img2 = cap2.read()
    succes3, img3 = cap3.read()
    
    start = time.perf_counter()
    
    stitched_image = imageStitching(img1,img2,img3)
    
    end = time.perf_counter()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: ", fps)

    cv2.putText(stitched_image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    cv2.imshow('Stitched Image', stitched_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    

# Release and destroy all windows before termination
cap1.release()
cap2.release()
cap3.release()

cv2.destroyAllWindows()"""
  
        