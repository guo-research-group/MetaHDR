import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.ticker import MaxNLocator


COEFFICIENTS = np.array([2.547644235590665,1,0.5264489489365198,0.24634900501576648,0.1341156465875588,0.06195803495800445,0.02994426928513942,0.01549933753090596,0.004225479869806865])


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

    # find homography
    src_pts = np.float32(
        [pts1[m.queryIdx] + pt1_st for m, n in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [pts2[m.trainIdx] + pt2_st for m, n in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    return M


def getAlignedImages(img):
    img1 = cv2.imread('./Imagedata/Calibration/Homography/1.tiff')[:, :, 0]
    img2 = cv2.imread('./Imagedata/Calibration/Homography/2.tiff')[:, :, 0]

    # key point of I1
    x_st1 = 970
    x_ed1 = 1550

    y_st1 = 670
    y_ed1 = 1240

    I1 = img1[y_st1:y_ed1, x_st1:x_ed1]
    pts1 = key_pts(I1, 50, 25)

    # key points of I2
    x_st2 = 311
    y_st2 = 132
    x_ed2 = 895
    y_ed2 = 700

    I2 = img2[y_st2:y_ed2, x_st2:x_ed2]
    pts2 = key_pts(I2, 37, 20)


    # key points of I3
    x_st3 = 320
    y_st3 = 672
    x_ed3 = 889
    y_ed3 = 1225

    I3 = img2[y_st3:y_ed3, x_st3:x_ed3]
    pts3 = key_pts(I3, 23, 15)


    # key points of I4
    x_st4 = 320
    y_st4 = 1230
    x_ed4 = 889
    y_ed4 = 1789


    I4 = img2[y_st4:y_ed4, x_st4:x_ed4]
    pts4 = key_pts(I4, 10, 5)


    # key points of I5
    x_st5 = 989
    y_st5 = 1218
    x_ed5 = 1540
    y_ed5 = 1789

    I5 = img2[y_st5:y_ed5, x_st5:x_ed5]
    pts5 = key_pts(I5, 7, 5)

    # key points of I7
    x_st7 = 1640
    y_st7 = 689
    x_ed7 = 2190
    y_ed7 = 1230


    I7 = img2[y_st7:y_ed7, x_st7:x_ed7]
    pts7 = key_pts(I7, 5, 2)

    # key points of I8
    x_st8 = 1610
    y_st8 = 136
    x_ed8 = 2191
    y_ed8 = 673

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
    H17 = homography(pts1, pts7, I1, I7, np.array(
        [x_st1, y_st1]), np.array([x_st7, y_st7]))
    
    H18 = homography(pts1, pts8, I1, I8, np.array(
        [x_st1, y_st1]), np.array([x_st8, y_st8]))

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
    
    return np.array([I1,I2_aligned,I3_aligned,I4_aligned,I5_aligned,I6_aligned,I7_aligned,I8_aligned,I9_aligned])


def getGradient_x(I):
    sobelx = np.gradient(I,axis=2)
    sobelx = np.where(I<255,sobelx,0)
    return sobelx


def getGradient_y(I):
    sobely = np.gradient(I,axis=1)
    sobely = np.where(I<255,sobely,0)
    return sobely


def frankotchellappa(del_f_del_x, del_f_del_y, reflec_pad=False):

    from numpy.fft import fft2, ifft2, fftfreq

    if reflec_pad:
        del_f_del_x, del_f_del_y = _reflec_pad_grad_fields(del_f_del_x,
                                                           del_f_del_y)
        
    NN, MM = del_f_del_x.shape[0:2]
    wx, wy = np.meshgrid(fftfreq(MM) * 2 * np.pi,
                         fftfreq(NN) * 2 * np.pi, indexing='xy')
    # by using fftfreq there is no need to use fftshift

    numerator = -1j * wx * fft2(del_f_del_x) - 1j * wy * fft2(del_f_del_y)

    denominator = (wx) ** 2 + (wy) ** 2 + np.finfo(float).eps

    res = ifft2(numerator / denominator)
    res -= np.mean(np.real(res))

    if reflec_pad:
        return _one_forth_of_array(res)
    else:
        return res


def _reflec_pad_grad_fields(del_func_x, del_func_y):

    del_func_x_c1 = np.concatenate((del_func_x,
                                    del_func_x[::-1, :]), axis=0)

    del_func_x_c2 = np.concatenate((-del_func_x[:, ::-1],
                                    -del_func_x[::-1, ::-1]), axis=0)

    del_func_x = np.concatenate((del_func_x_c1, del_func_x_c2), axis=1)

    del_func_y_c1 = np.concatenate((del_func_y,
                                    -del_func_y[::-1, :]), axis=0)

    del_func_y_c2 = np.concatenate((del_func_y[:, ::-1],
                                    -del_func_y[::-1, ::-1]), axis=0)

    del_func_y = np.concatenate((del_func_y_c1, del_func_y_c2), axis=1)

    return del_func_x, del_func_y


def _one_forth_of_array(array):

    array, _ = np.array_split(array, 2, axis=0)
    return np.array_split(array, 2, axis=1)[0]


def _grad(func):

    del_func_2d_x = np.diff(func, axis=1)
    del_func_2d_x = np.pad(del_func_2d_x, ((0, 0), (1, 0)), 'edge')

    del_func_2d_y = np.diff(func, axis=0)
    del_func_2d_y = np.pad(del_func_2d_y, ((1, 0), (0, 0)), 'edge')

    return del_func_2d_x, del_func_2d_y


def getEstimatedGradient(img,G,t):
    
    I = getAlignedImages(img)

    eta = 0.68

    I_gradient_x = getGradient_x(I)
    I_gradient_y = getGradient_y(I)

    alpha = COEFFICIENTS[:, np.newaxis, np.newaxis] * np.ones_like(I[0])

    M = (I != 255).astype(np.float64)

    E_star_gradient_x = np.sum(alpha[:9]*I_gradient_x[:9]*M[:9],axis=0)/np.sum(eta*t*alpha[:9]**2*M[:9],axis=0)/G
    E_star_gradient_y = np.sum(alpha[:9]*I_gradient_y[:9]*M[:9],axis=0)/np.sum(eta*t*alpha[:9]**2*M[:9],axis=0)/G

    LDR_energy = np.sum(I[4] * (I[4] != 255))/COEFFICIENTS[4]

    return E_star_gradient_x,E_star_gradient_y,LDR_energy


def getEstar(img,G,t):

    Ex,Ey,LDR_energy = getEstimatedGradient(img,G,t)

    E_star = frankotchellappa(Ex,Ey)

    E_star = np.real(E_star)

    energy_ratio = LDR_energy/np.sum(np.abs(E_star))

    E_star *= energy_ratio

    return E_star


def getTonemapped(img,G,t):

    E_star = getEstar(img,G,t)

    E_star = np.stack((E_star,E_star,E_star), axis=-1)

    tonemapDrago = cv2.createTonemapDrago(1, 0.7)
    ldrDrago = tonemapDrago.process(E_star.astype("float32"))

    tonemapReinhard = cv2.createTonemapReinhard(0.7, 0,1,0)
    ldrReinhard = tonemapReinhard.process(E_star.astype("float32"))

    tonemapMantiuk = cv2.createTonemapMantiuk(1,0.7, 1)
    ldrMantiuk = tonemapMantiuk.process(E_star.astype("float32"))

    return ldrReinhard


img = cv2.imread("./Imagedata/Reconstruction/watchgear.tiff",0)

I = getAlignedImages(img)

tonemapped = getTonemapped(img,1,1.5)

fig,ax = plt.subplots(1,2)
ax[0].imshow(I[0,:,:],cmap='gray')
ax[0].set_title('LDR image')

ax[1].imshow(tonemapped,cmap='gray')
ax[1].set_title('Tonemapped HDR image')
plt.show()