import numpy as np
import cv2

def key_pts(I1, b1, b2):
    I1 = cv2.medianBlur(I1, 11)
    circles = cv2.HoughCircles(I1, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=b1, param2=b2,
                               minRadius=10, maxRadius=15)
    # edges = cv2.Canny(I1, 50, 10)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(I1, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(I1, center, radius, (255, 0, 255), 3)

    # fig = plt.figure()
    # plt.imshow(I1)
    # plt.show()

    return circles[0, :, 0:2]


def homography(pts1, pts2, img1, img2, pt1_st, pt2_st):
    # find the matching using K nearest neighbors
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(pts1.astype(np.float32),
                             pts2.astype(np.float32), k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1, [cv2.KeyPoint(pts1[i, 0], pts1[i, 1], 1) for i in range(pts1.shape[0])], img2, [cv2.KeyPoint(pts2[i, 0], pts2[i, 1], 1) for i in range(pts2.shape[0])],
                              matches, None, **draw_params)
    # fig = plt.figure(figsize=(6, 3))
    # plt.imshow(img3,)
    # plt.axis('off')
    # plt.show()

    # find homography
    src_pts = np.float32(
        [pts1[m.queryIdx] + pt1_st for m, n in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [pts2[m.trainIdx] + pt2_st for m, n in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # # warp images
    # img2_warped = cv2.warpPerspective(img2, M, img1.shape[::-1])

    return M


def getAlignedImages(img):
    img1 = cv2.imread('./Imagedata/Calibration/Homography/1.tiff')[:, :, 0]
    img2 = cv2.imread('./Imagedata/Calibration/Homography/2.tiff')[:, :, 0]

    # key point of I1
    x_st1 = 950
    x_ed1 = 1530

    y_st1 = 650
    y_ed1 = 1220

    I1 = img1[y_st1:y_ed1, x_st1:x_ed1]
    pts1 = key_pts(I1, 50, 25)

    # key points of I2
    x_st2 = 291
    y_st2 = 112
    x_ed2 = 875
    y_ed2 = 680

    I2 = img2[y_st2:y_ed2, x_st2:x_ed2]
    pts2 = key_pts(I2, 37, 20)


    # key points of I3
    x_st3 = 300
    y_st3 = 652
    x_ed3 = 869
    y_ed3 = 1205


    I3 = img2[y_st3:y_ed3, x_st3:x_ed3]
    pts3 = key_pts(I3, 23, 15)


    # key points of I4
    x_st4 = 300
    y_st4 = 1210
    x_ed4 = 869
    y_ed4 = 1769


    I4 = img2[y_st4:y_ed4, x_st4:x_ed4]
    pts4 = key_pts(I4, 10, 5)


    # key points of I5
    x_st5 = 969
    y_st5 = 1198
    x_ed5 = 1520
    y_ed5 = 1769


    I5 = img2[y_st5:y_ed5, x_st5:x_ed5]
    pts5 = key_pts(I5, 7, 5)

    # key points of I7
    x_st7 = 1620
    y_st7 = 669
    x_ed7 = 2170
    y_ed7 = 1210


    I7 = img2[y_st7:y_ed7, x_st7:x_ed7]
    pts7 = key_pts(I7, 5, 2)

    # key points of I8
    x_st8 = 1590
    y_st8 = 116
    x_ed8 = 2171
    y_ed8 = 653


    I8 = img2[y_st8:y_ed8, x_st8:x_ed8]
    pts8 = key_pts(I8, 5, 2)


    # find matching points
    H12 = homography(pts1, pts2, I1, I2, np.array(
        [x_st1, y_st1]), np.array([x_st2, y_st2]))
    H13 = homography(pts1, pts3, I1, I3, np.array(
        [x_st1, y_st1]), np.array([x_st3, y_st3]))
    H14 = homography(pts1, pts4, I1, I4, np.array(
        [x_st1, y_st1]), np.array([x_st4, y_st4]))
    H15 = homography(pts1, pts5, I1, I5, np.array(
        [x_st1, y_st1]), np.array([x_st5, y_st5]))
    H16 = np.linalg.inv(H12)
    # H17 = np.linalg.inv(H13)
    H17 = homography(pts1, pts7, I1, I7, np.array(
        [x_st1, y_st1]), np.array([x_st7, y_st7]))
    
    H18 = homography(pts1, pts8, I1, I8, np.array(
        [x_st1, y_st1]), np.array([x_st8, y_st8]))

    # H18 = H12 @ np.linalg.inv(H13) @ np.linalg.inv(H13)
    H19 = H12 @ np.linalg.inv(H13)

    I1 = img[y_st1:y_ed1, x_st1:x_ed1]

    I2_aligned = cv2.warpPerspective(
        img, H12, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I3_aligned = cv2.warpPerspective(
        img, H13, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I4_aligned = cv2.warpPerspective(
        img, H14, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I5_aligned = cv2.warpPerspective(
        img, H15, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I6_aligned = cv2.warpPerspective(
        img, H16, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I7_aligned = cv2.warpPerspective(
        img, H17, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I8_aligned = cv2.warpPerspective(
        img, H18, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    I9_aligned = cv2.warpPerspective(
        img, H19, img1.shape[::-1])[y_st1:y_ed1, x_st1:x_ed1]
    
    return I1,I2_aligned,I3_aligned,I4_aligned,I5_aligned,I6_aligned,I7_aligned,I8_aligned,I9_aligned


def getGradient(I):
    _,sobelx,sobely = np.gradient(I)
    return np.hstack((sobelx,sobely))


####################################
######### Calibration ##############
####################################

# We prepare 10 textures and there are totally 100 imgaes in each of them
num_texture = np.arange(1,11)
t = np.arange(100)


I1_img = []
I2I3_img = []
I4I9_img = []

gradient_matrix_I = []

for j in num_texture:
    for i in t:
        # the "true" image I2, and I3, are from exposure time t = 250000
        img = cv2.imread(f"./Imagedata/Calibration/Texture/texture{j}/250000/{i}.tiff",0)
        I2I3_img.append(img)

        # I1 is from exposure time t = 50000
        img = cv2.imread(f"./Imagedata/Calibration/Texture/texture{j}/50000/{i}.tiff",0)
        I1_img.append(img)

        # I4 - I9 are from exposure time t = 500000
        img = cv2.imread(f"./Imagedata/Calibration/Texture/texture{j}/500000/{i}.tiff",0)
        I4I9_img.append(img)


    I1_img = np.array(I1_img)
    I2I3_img = np.array(I2I3_img)
    I4I9_img = np.array(I4I9_img)

    # Take the average to eliminate the noise
    I1_img = np.mean(I1_img,axis=0)
    I2I3_img = np.mean(I2I3_img,axis=0)
    I4I9_img = np.mean(I4I9_img,axis=0)
    
    I1,_,_,_,_,_,_,_,_ = getAlignedImages(I1_img)
    _,I2,I3,_,_,_,_,_,_ = getAlignedImages(I2I3_img)
    _,_,_,I4,I5,I6,I7,I8,I9 = getAlignedImages(I4I9_img)

    # We need to do the calibration under the same camera setting. The camera is nearly linear, which allows us to multiply the image by the exposure ratio
    I1 *= 10
    I2 *= 2
    I3 *= 2

    I = np.stack((I1,I2,I3,I4,I5,I6,I7,I8,I9))

    I1_img = []
    I2I3_img = []
    I4I9_img = []

    gradient_matrix_I.append(getGradient(I))

gradient_matrix_I = np.array(gradient_matrix_I)

# texture number * number of gradient directions (along x and y direction) * image size
N = 10*2*570*580

new_I1 = gradient_matrix_I[:,0,:,:].reshape((N,1))
new_I2 = gradient_matrix_I[:,1,:,:].reshape((N,1))
new_I3 = gradient_matrix_I[:,2,:,:].reshape((N,1))
new_I4 = gradient_matrix_I[:,3,:,:].reshape((N,1))
new_I5 = gradient_matrix_I[:,4,:,:].reshape((N,1))
new_I6 = gradient_matrix_I[:,5,:,:].reshape((N,1))
new_I7 = gradient_matrix_I[:,6,:,:].reshape((N,1))
new_I8 = gradient_matrix_I[:,7,:,:].reshape((N,1))
new_I9 = gradient_matrix_I[:,8,:,:].reshape((N,1))


# least square solution
a12 = np.linalg.lstsq(new_I2,new_I1,rcond=None)[0][0][0]
a32 = np.linalg.lstsq(new_I2,new_I3,rcond=None)[0][0][0]
a42 = np.linalg.lstsq(new_I2,new_I4,rcond=None)[0][0][0]
a52 = np.linalg.lstsq(new_I2,new_I5,rcond=None)[0][0][0]
a62 = np.linalg.lstsq(new_I2,new_I6,rcond=None)[0][0][0]
a72 = np.linalg.lstsq(new_I2,new_I7,rcond=None)[0][0][0]
a82 = np.linalg.lstsq(new_I2,new_I8,rcond=None)[0][0][0]
a92 = np.linalg.lstsq(new_I2,new_I9,rcond=None)[0][0][0]

print(a12)
print("1")
print(a32)
print(a42)
print(a52)
print(a62)
print(a72)
print(a82)
print(a92)
