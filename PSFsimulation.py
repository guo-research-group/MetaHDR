import numpy as np
import tensorflow as tf
from scipy import signal
from PIL import Image
import matplotlib.pyplot as plt
import math
import scipy.io

import dflat.optimization_helpers as df_optimizer
import dflat.fourier_layer as df_fourier
import dflat.neural_optical_layer as df_neural
import dflat.physical_optical_layer as df_physical
import dflat.data_structure as df_struct
import dflat.tools as df_tools


def fft_convolve_zero_padding_2d(
        images,  # (N, image_shape[0], image_shape[1], channel)
        psfs,  # (N, psf_shape[0], psf_shape[1], channel)
        parameters,
        mode='same',
        sum_channel=False):
        
    # the channel will be summed before the ifft
    dtype = parameters['dtype']
    images = tf.cast(images, dtype=dtype)
    psfs = tf.cast(psfs, dtype=dtype)

    # make the psf and image in the same shape by padding
    # the convolution shape is the sum of both shapes
    # so that the convolution is zero padding
    image_shape = images.shape[1:-1]
    psf_shape = psfs.shape[1:-1]

    images = tf.image.resize_with_crop_or_pad(
        images,
        int(image_shape[0]) + int(psf_shape[0]) - 1,
        int(image_shape[1]) + int(psf_shape[1]) - 1)

    psfs = tf.image.resize_with_crop_or_pad(
        psfs,
        int(image_shape[0]) + int(psf_shape[0]) - 1,
        int(image_shape[1]) + int(psf_shape[1]) - 1)

    # then use fft convolution for each pinhole image
    # fourier centering
    images_transpose = tf.transpose(images, [0, 3, 1, 2])
    psfs_transpose = tf.transpose(psfs, [0, 3, 1, 2])
    if (dtype is tf.complex128) or (dtype is tf.complex64):
        F_images_transpose = tf.signal.fft2d(images_transpose)
        F_psfs_transpose = tf.signal.fft2d(psfs_transpose)
    else:
        F_images_transpose = tf.signal.fft2d(
            tf.complex(images_transpose, tf.cast(0, dtype=dtype)))
        F_psfs_transpose = tf.signal.fft2d(
            tf.complex(psfs_transpose, tf.cast(0, dtype=dtype)))
    F_blurry_images_transpose = F_images_transpose * F_psfs_transpose

    # needs to be centered before transforming back
    # the center factor is (N-1)/N for odd shape
    # according to
    # https://dsp.stackexchange.com/questions/38746/centered-fourier-transform
    x_center_factor = (tf.shape(F_blurry_images_transpose)[3] -
                       1) / tf.shape(F_blurry_images_transpose)[3]
    x = tf.range(tf.shape(F_blurry_images_transpose)[3])
    x = tf.cast(x, dtype=dtype)
    x_center_factor = tf.cast(x_center_factor, dtype=dtype)
    x *= x_center_factor

    y_center_factor = (tf.shape(F_blurry_images_transpose)[2] -
                       1) / tf.shape(F_blurry_images_transpose)[2]
    y = tf.range(tf.shape(F_blurry_images_transpose)[2])
    y = tf.cast(y, dtype=dtype)
    y_center_factor = tf.cast(y_center_factor, dtype=dtype)
    y *= y_center_factor

    xx, yy = tf.meshgrid(x, y)
    xx = tf.cast(tf.expand_dims(tf.expand_dims(xx, 0), 0), dtype=dtype)
    yy = tf.cast(tf.expand_dims(tf.expand_dims(yy, 0), 0), dtype=dtype)

    # NOTICE: we do not use convenient variable because
    # we once met the problem that -1-j0 not equal to -1+j0
    fourier_centering_term = tf.cast(
        -1, dtype=F_blurry_images_transpose.dtype)**tf.cast(
            xx + yy + 1, dtype=F_blurry_images_transpose.dtype)
    if sum_channel:
        F_blurry_images_transpose = tf.reduce_sum(F_blurry_images_transpose,
                                                  axis=1,
                                                  keepdims=True)
    blurry_images_transpose = -tf.signal.ifft2d(
        F_blurry_images_transpose * fourier_centering_term
    )  # (N, channel, padded_shape[0], padded_shape[1])

    if (dtype is not tf.complex128) and (dtype is not tf.complex64):
        blurry_images_transpose = tf.abs(blurry_images_transpose)

    blurry_images = tf.transpose(blurry_images_transpose, [0, 2, 3, 1])
    if mode == 'same':
        blurry_images = tf.image.resize_with_crop_or_pad(
            blurry_images, int(image_shape[0]), int(image_shape[1]))

    return blurry_images  #(N, image_shape[0], image_shape[1], 1 or channel)


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    currshape = tf.shape(arr)[-2:]
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    return arr[..., startind[0]:endind[0], startind[1]:endind[1]]

def fftconv(in1, in2, mode="full"):
    # Reorder channels to come second (needed for fft)
    in1 = tf.transpose(in1, perm=[0, 3, 1, 2])
    in2 = tf.transpose(in2, perm=[0, 3, 1, 2])

    # Extract shapes
    s1 = tf.convert_to_tensor(tf.shape(in1)[-2:])
    s2 = tf.convert_to_tensor(tf.shape(in2)[-2:])
    shape = s1 + s2 - 1

    # Compute convolution in fourier space
    sp1 = tf.signal.rfft2d(in1, shape)
    sp2 = tf.signal.rfft2d(in2, shape)
    ret = tf.signal.irfft2d(sp1 * sp2, shape)

    # Crop according to mode
    if mode == "full":
        cropped = ret
    elif mode == "same":
        cropped = _centered(ret, s1)
    elif mode == "valid":
        cropped = _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

    # Reorder channels to last
    result = tf.transpose(cropped, perm=[0, 2, 3, 1])
    return result

def calculate_SNR(N_signal,GT_signal):
    """
    Computes the SNR of the Noisy signal by extracting the noise component from
    the Noisy signal via the GT signal:
    E.g.) Noise = N_signal - GT_signal
    """
    # extract the noise component
    Noise = N_signal - GT_signal
    # compute expected values using mean and variance
    ExS2 = np.var(N_signal) + np.mean(N_signal)**2
    ExN2 = np.var(Noise) + np.mean(Noise)**2

    SNR = 10*np.log10(ExS2/ExN2) # SNR in dB
    return(SNR)


def calculate_PSNR(signal,ref):
    """
    Computes the PSNR of a signal compared to a reference
    """
    maxI = np.max(ref)
    MSE = np.sum((signal-ref)**2)/np.size(signal)

    PSNR = 10*np.log10((maxI**2)/MSE)

    return(PSNR)

def calculate_MSE(signal,ref):
    """
    Computes the MSE of the signal compared to the reference
    """
    MSE = np.sum((signal-ref)**2)/np.size(signal)
    return(MSE)


def gen_delta_target(row,col,x_locs,y_locs,weights):
    # generates a target image/psf that places delta functions at specified locations, with specified weights
    target = np.zeros((row,col))
    for k in range(len(x_locs)):
        if y_locs[k]<=2048:
            target[y_locs[k]][x_locs[k]] = weights[k]
        else:
            target[2048][x_locs[k]] = 0
    return target


# Function that converts a phase and transmittence profile to a metasurface physical profile
def lookup_D1_pol1(phaseTable, transTable, p1_vect, wavelength, use_wavelength, ms_trans, ms_phase):

    # Find the sub-table matching the wavelength requested
    _, w_idx = min((val, idx) for (idx, val) in enumerate(np.abs(use_wavelength - wavelength)))
    sublib_phase = phaseTable[w_idx, :]
    sublib_trans = transTable[w_idx, :]
    or_table = sublib_trans * np.exp(1j * sublib_phase)
    print("w_idx: ",w_idx)

    # Get the target profile
    target_profile = ms_trans * np.exp(1j * ms_phase)

    # minimize via look-up table
    row = ms_phase.shape[0]
    col = ms_phase.shape[1]

    design_surface = np.zeros((row,col))
    for i in range(row):
        for j in range(col):
            target = target_profile[i][j]
            _, param_idx = min((val, idx) for (idx, val) in enumerate(np.abs(or_table- target)))
            design_surface[i][j] = p1_vect[param_idx]
            
    # Define a normalized shape vector for convenience
    design_surface = np.array(design_surface)

    return design_surface

# function that converts the surface profile into a phase and transmittence profile
def surface_to_profiles(surface_profile,radii,wavelength,lib_phase,lib_trans,active_wvl):
    # takes the physical metasurface and a library of phase and transmittence and returns the surface phase and transmittence profiles
    phase_from_surface = np.zeros(np.shape(surface_profile))
    trans_from_surface = np.zeros(np.shape(surface_profile))

    for i in range(np.size(surface_profile,0)):
        for j in range(np.size(surface_profile,1)):
            wvl_idx = [z for z, val in enumerate(wavelength == active_wvl) if val]
            radii_idx = [z for z, val in enumerate(radii == surface_profile[i][j]) if val]
            phase_from_surface[i][j] = lib_phase[wvl_idx[0]][radii_idx[0]]
            trans_from_surface[i][j] = lib_trans[wvl_idx[0]][radii_idx[0]]

    return np.expand_dims(phase_from_surface,0),np.expand_dims(trans_from_surface,0)




if __name__ == "__main__":

    pixel2cell = 4

    # Define propagation parameters for psf calculation
    propagation_parameters = df_struct.prop_params(
        {
            "wavelength_m": 650e-9,
            "ms_length_m": {"x": 1.0e-3, "y": 1.0e-3},
            "ms_dx_m": {"x": 300e-9*pixel2cell, "y": 300e-9*pixel2cell},
            "radius_m": 1.0e-3 / 2.01,
            "sensor_distance_m": 5e-2,
            "initial_sensor_dx_m": {"x": 3.45e-6, "y": 3.45e-6},
            "sensor_pixel_size_m": {"x": 3.45e-6, "y": 3.45e-6},
            "sensor_pixel_number": {"x": 2449, "y": 2049}, 
            "radial_symmetry": False,
            "diffractionEngine": "fresnel_fourier",
            "accurate_measurement": False,  # Flag ensures output grid is exact but is expensive
        },
        verbose=False,
    )

    ######## ----------------------------------------------------------------------------------------------------------------- ########
    ######## ------------------------------------------------ ideal Lens Profiles -------------------------------------------- ########
    ######## ----------------------------------------------------------------------------------------------------------------- ########
    # Point_source locs we want to compute the psf for
    point_source_locs = np.array([[0.0, 0.0, 1e6]])  # on-axis ps at 1e6 m away (~infinity)
    sensor_pixel_number = propagation_parameters["sensor_pixel_number"]
    cidx_y = sensor_pixel_number["y"]//2 
    cidx_x = sensor_pixel_number["x"]//2


    # define the location of the shifted focal points (9-multiplex)
    x_locs = [cidx_x//2,cidx_x,cidx_x*3//2,cidx_x//2,cidx_x,cidx_x*3//2,cidx_x//2,cidx_x,cidx_x*3//2]
    y_locs = [cidx_y//2,cidx_y//2,cidx_y//2,cidx_y,cidx_y,cidx_y,cidx_y*3//2,cidx_y*3//2,cidx_y*3//2]
    weights = [1/(2**4),1/(2**5),1/(2**6),1/(2**2),1/(2**1),1/(2**7),1/(2**2),1/(2**9),1/(2**8)]

    # generate a map of the distribution we want to draw out samples from
    col = propagation_parameters["calc_samplesM"]["x"]
    row = propagation_parameters["calc_samplesM"]["y"]
    prob = np.array(weights)**(1/2)
    idx = np.random.choice(9,(row,col),p=prob/np.sum(prob))

    # pre alocate the phase and transmitence profiles 
    interleave_phase = np.zeros((1,row,col))
    interleave_trans = np.zeros((1,row,col))

    for k in range(len(x_locs)):
        # determine the focus shifting amount
        x_shift = x_locs[k] - cidx_x
        y_shift = y_locs[k] - cidx_y

        # Dflat function that generates an idea focusing phase and transmittence profile
        focus_trans, focus_phase, ap_trans, _ = df_fourier.focus_lens_init(
            propagation_parameters, 
            [650e-9], 
            [1e6], 
            [{"x": x_shift*propagation_parameters["sensor_pixel_size_m"]["x"], "y": y_shift*propagation_parameters["sensor_pixel_size_m"]["y"]}]
        )
        interleave_phase[0][idx==k] = focus_phase[0][idx==k]
        printout = "Completed interleaving of focus" + str(k+1)
        printout2 = 'Number of cells for focus ' + str(k+1) + '= ' + str(np.sum(idx==k))
        print(printout) 
        print(printout2)
    print(prob/np.sum(prob))
    print(np.sum(ap_trans[ap_trans>0.1]))
    
    ######## ----------------------------------------------------------------------------------------------------------------- ########
    ######## ------------------------------------------------ Surface Profile ------------------------------------------------ ########
    ######## ----------------------------------------------------------------------------------------------------------------- ########
    # Now that we have the interleaved lens profile construct a physical metasurface profile
    # load the library and the lens profile
    library_data = scipy.io.loadmat("./Metasurface_Library/AmorphSi_U300nm_H300nm.mat")
    wavelength_list = library_data["w"]
    radii = library_data["radii"]
    phase_300 = np.transpose(library_data['phase_profileX'])
    trans_300 = np.transpose(library_data['trans_profileX'])
    # use the library to generate a metasurface shape profile
    wavelength = 650e-9
    surface_matrix = lookup_D1_pol1(phase_300, trans_300, radii, wavelength, wavelength_list, ap_trans[0], interleave_phase[0])# do not change the first wavelength

    ######## ----------------------------------------------------------------------------------------------------------------- ########
    ######## ---------------------------------------------- Real lens profiles ----------------------------------------------- ########
    ######## ----------------------------------------------------------------------------------------------------------------- ########
    
    real_phase, real_trans = surface_to_profiles(surface_matrix,radii,wavelength_list,phase_300,trans_300,propagation_parameters["wavelength_m"])# change the wavelength here

    # Use the ideal focusing function to generate an aperture 
    _, _, ap_trans, _ = df_fourier.focus_lens_init(
        propagation_parameters, 
        [650e-9], 
        [1e6], 
        [{"x": 0, "y":0}]
    )
     
    z_loc = 1e3 # point source location
    
    # Identify the PSF of the interleaved phase profile
    
    theta = 0 # Simulate on/off-axis angle

    psf_comp = df_fourier.PSF_Layer_Mono(propagation_parameters) # change the wavelength here
    psf_intensity,psf_phase = psf_comp(inputs=[ap_trans*real_trans,real_phase],point_source_locs=[[0,z_loc*np.tan(theta/180*np.pi),z_loc]])
    interleave_psf = psf_intensity.numpy()[0,0,0,:,:]
    interleave_phazor = psf_intensity.numpy()*np.exp(1j*psf_phase.numpy())
    
    np.save('interleaved_psf',interleave_psf)

    ######### 0th Diffraction order PSF generation #########
    residual_trans = ap_trans - ap_trans*real_trans
    psf_intensity,psf_phase = psf_comp(inputs=[residual_trans,np.zeros((1,row,col))],point_source_locs=[[0,z_loc*np.tan(theta/180*np.pi),z_loc]])
    residual_psf = psf_intensity.numpy()[0,0,0,:,:]
    residual_phazor = psf_intensity.numpy()*np.exp(1j*psf_phase.numpy())
    

    ######### Final PSF generation ######### 
    final_phazor = interleave_phazor + residual_phazor 
    final_psf = np.abs(final_phazor[0,0,0,:,:]) 

    
    ######## ----------------------------------------------------------------------------------------------------------------- ########
    ######## ------------------------------------------- Examine Image components -------------------------------------------- ########
    ######## ----------------------------------------------------------------------------------------------------------------- ########

    # generate a circle to examine the energy ratio 
    x_line = np.linspace(-1,1,500)
    x_grid,y_grid = np.meshgrid(x_line,x_line)
    circle = np.array((x_grid**2+y_grid**2 < 0.8),dtype=np.float32)

    circle = np.array(Image.open("./Imagedata/Simulation/46.gif"),dtype=np.float32)/255
    
    Img = signal.fftconvolve(final_psf,circle,mode='same')
    Img_ideal = signal.fftconvolve(interleave_psf,circle,mode='same')
    Img_residual = signal.fftconvolve(residual_psf,circle,mode='same')

    # examine the PSNR
    ## define path for images 
    img_path = "./Imagedata/Simulation"
    interleave_psf = interleave_psf/np.sum(interleave_psf)
    interleave_psf = np.flipud(interleave_psf)
    
    # generate a target image
    row_sen = interleave_psf.shape[0]
    col_sen = interleave_psf.shape[1]
    
    target_im = gen_delta_target(row_sen,col_sen,x_locs,y_locs,weights)
    target_im1 = target_im/np.sum(target_im)

    ######## Obtain the average PSNR for the noise model under various conditions ########
    PSNR = 0
    for j in range(17): # We have 17 distinct images involved in the calculation of PSNR
        file_path  = img_path + "/" + str(j+31) +'.gif'
        img = np.array(Image.open(file_path)).astype(np.float32)
        # obtain the image
        final_psf = scipy.signal.fftconvolve(interleave_psf,img,'same')
        target_im = scipy.signal.fftconvolve(target_im1,img,'same')

        # crop the brightest and dimmest image
        I9_final_psf = final_psf[sensor_pixel_number["y"]-y_locs[7]-256:sensor_pixel_number["y"]-y_locs[7]+256,x_locs[7]-256:x_locs[7]+256]
        
        # normalize the image
        I9_final_psf = (I9_final_psf-np.min(I9_final_psf))/(np.max(I9_final_psf)-np.min(I9_final_psf))
        img = (img-np.min(img))/(np.max(img)-np.min(img))

        # calculate the PSNR between groundtruth and the dimmest image
        PSNR += calculate_PSNR(I9_final_psf,img)
    print('-----------------------------------------------------')
    print('PSNR in dB for multiplex: ',PSNR/17)

    plt.figure()
    plt.imshow(np.log(interleave_psf),cmap='jet')
    plt.axis('off')

    plt.figure()
    plt.imshow(np.log(final_psf),cmap='gray')
    plt.colorbar()
    plt.axis('off')

    plt.figure()
    plt.imshow(final_psf,cmap='gray')
    plt.axis('off')
    plt.show()
