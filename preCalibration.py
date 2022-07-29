import cv2
import numpy as np
import sys
import glob
import imutils
import matplotlib.pyplot as plt
import time


class Image_Stitching():
    
    def __init__(self):
        self.ratio=0.7
        self.min_match=30
        self.sift=cv2.SIFT_create()
        #self.sift=cv2.ORB_create()
        self.smoothing_window_size=200
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def registration(self, img1, img2):
        
        # Detect keypoints and compute descriptors
        #img1 = cv2.convertScaleAbs(img1)
        #img2 = cv2.convertScaleAbs(img2)
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        #matcher = cv2.BFMatcher()
        #raw_matches = matcher.knnMatch(des1, des2, k=2)
        raw_matches = self.flann.knnMatch(des1, des2, k=2)
        
        # Filter matches using the Lowe's ratio test
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        
        
        # Estimate homography matrix
        if len(good_matches) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,4.0)
            
            print(H)
                
            return H
        return None
    
    
    def create_mask(self,img1,img2,version):
        
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        # Create mask
        if version == 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
            
        return cv2.merge([mask, mask, mask])
    
    def Convert_xy(self, x, y):
        global center, f

        xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
        yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
        
        return xt, yt
    
    def projectOntoCylinder(self, InitialImage):
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
        ii_x, ii_y = self.Convert_xy(ti_x, ti_y)

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

    
    
    def image_rectification(self, img1, img2):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Visualize keypoints
        #imgSift = cv2.drawKeypoints(
            #img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("SIFT Keypoints", imgSift)

        #cv2.waitKey(0)

        # Match keypoints in both images
        index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)


        # Keep good matches: calculate distinctive image features
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < self.ratio*n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)


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

        

    def blending(self, img1, img2, H):
            
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        mask1 = self.create_mask(img1,img2,version='left_image')
        #cv2.imshow("mask1", mask1)
        #cv2.waitKey(0)
        
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        mask2 = self.create_mask(img1,img2,version='right_image')
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
    def trim(self, img3):
 
        grayimage = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(grayimage,5,255,cv2.THRESH_BINARY)      
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            crop = img3[y:y+h,x:x+w]
        return crop
    
    
    
def imageStitching():
    
    
    imageStitcher = Image_Stitching()
    imageStitcher.smoothing_window_size = 50

    #img1_rectified, img2_rectified, pts1, pts2, good_matches = imageStitcher.image_rectification(img1, img2)
    #H = imageStitcher.registration(img1, img2)
    #final1 = imageStitcher.blending(img1, img2, H)
    #final1 = cv2.convertScaleAbs(final1)
    #final1 = imageStitcher.trim(final1)
    #cv2.imwrite('panorama1.png', final1)
    #final1,_,_ = imageStitcher.projectOntoCylinder(final1)
    #cv2.imwrite('panorama1Cyl.png', final1)
    #cv2.imshow("Panorama", final1)
    #cv2.waitKey(0)
    

    
    #return final1


    

    
image_paths = sorted(glob.glob('C:/Users/nhoei/knightec/newImages/second/*.png'))
first_images = []

for image in image_paths:
    img = cv2.imread(image)
    first_images.append(img)
    
image_paths = sorted(glob.glob('C:/Users/nhoei/knightec/newImages/third/*.png'))
second_images = []

for image in image_paths:
    img = cv2.imread(image)
    second_images.append(img)




imageStitcher = Image_Stitching()
imageStitcher.smoothing_window_size = 50

f1,f2,f3 = [],[],[]
s1,s2,s3 = [],[],[]
t1,t2,t3 = [],[],[]

for i in range(len(first_images)):
    H = imageStitcher.registration(first_images[i], second_images[i])
    if H is not None:
        print(H)
        f1.append(H[0][0])
        f2.append(H[0][1])
        f3.append(H[0][2])
        s1.append(H[1][0])
        s2.append(H[1][1])
        s3.append(H[1][2])
        t1.append(H[2][0])
        t2.append(H[2][1])
        t3.append(H[2][2])


f1 = np.mean(np.array(f1))
f2 = np.mean(np.array(f2))
f3 = np.mean(np.array(f3))
s1 = np.mean(np.array(s1))
s2 = np.mean(np.array(s2))
s3 = np.mean(np.array(s3))
t1 = np.mean(np.array(t1))
t2 = np.mean(np.array(t2))
t3 = np.mean(np.array(t3))


Homography = np.array([[f1,f2,f3],[s1,s2,s3],[t1,t2,t3]])

print("H: ", Homography)

with open('homographyCalibrated.npy', 'wb') as f:
            np.save(f, np.array(H))
            

