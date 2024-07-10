# MetaHDR: Single Shot High-dynamic Range Imaging and Sensing Using a Multifunctional Metasurface

_Charles Brookshire*, Yuxuan Liu*, Yuanrui Chen, Wei-Ting Chen, and Qi Guo_

*Co-first authors with equal contribution

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: liu3910@purdue.edu


[Paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-15-26690&id=553173)
## Introduction

MetaHDR is a single-shot high-dynamic range (HDR) imaging and sensing system using a multifunctional metasurface. The system can simultaneously form 9 low dynamic range (LDR) images with distinct irradiance on a photosensor, and fuse them into HDR image using a gradient-based algorithm. MetaHDR achieves single-shot HDR photography and videography that increases the dynamic range by at least 50 dB compared to the original dynamic range of the photosensor. 

We divide the implementation of MetaHDR into two independnt parts, metasurface design and gradient-based HDR reconstruction. The relationship between the code and data in this repository is as follows:
```
MetaHDR
│   
│
└───Function: Metasurface design and simulation
    └───Code
        │   FDTD_simulation.lsf
        │   PSFsimulation.py
    └───Data
        └───Metasurface_Library
            │   AmorphSi_U300nm_H300nm.mat
        └───Imagedata
            └───Simulation
                │   31.gif
                │   ...
                │   47.gif

└───Function: HDR reconstruction
    └───Code
        │   calibration.py
        │   HDRreconstructioni.py
    └───Data
        └───Imagedata
            └───Reconstruction
                    │   circuit.tiff
                    │   watchgear.tiff
            └───Calibration
                └───Homography
                    │   1.tiff
                    │   2.tiff
                └───Texture
                    └───texture1
                        └───50000
                            │   0.tiff
                            │   ...
                            │   99.tiff
                        └───200000
                            │   0.tiff
                            │   ...
                            │   99.tiff
                        └───250000
                            │   0.tiff
                            │   ...
                            │   99.tiff
                        └───500000
                            │   0.tiff
                            │   ...
                            │   99.tiff
```

## Usage

### Installation
To install our code, run
```
git clone https://github.com/guo-research-group/MetaHDR
cd MetaHDR 
```

### FDTD Simulation

The Lumerical script  `FDTD_simulation.lsf` uses a nano-cell with 775nm height and 300nm X 300nm nano-cell size, and sweeps across 310 nm – 770 nm wavelengths of light.

### Metasurface Simulation

We designed and fabricated a 1 mm diameter multifunctional metasurface with focal length 5cm operating at wavelength 650nm. The designed power ratio between every two adjacent images is 2. To verify our design, we estimate the point spread function (PSF) of the designed metasurface using the D-Flat simulator. So make sure you have successfully installed DFlat before continuing.
```
git clone https://github.com/DeanHazineh/DFlat
pip install -e .
python PSFsimulation.py
```
In the simulatio code `PSFsimulation`, we model a uniform intensity point source @ $\lambda$ = 650 nm at a distance of 1 km and record the PSF arrangements. We also examine the average peak signal-to-noise ratio (PSNR) of synthesized images using the PSFs under the noise model in Eq. 7 for various realistic photon levels and noise levels. The simulated images are generated as the convolution of our simulated PSF and a 512×512 8-bit image, which can be found in `Imagedata->Simulation->*`

Run 
```
python PSFsimulation.py
```
to fully check the implementation.

### HDR reconstruction
The calibration of MetaHDR involves two steps: geometric alignment and contrast registration. The geometric alignment determines a homography $𝐻_{ij}$ between every pair of sub-images $𝐼_i(𝑥, 𝑦)$
and $𝐼_𝑗(𝑥, 𝑦)$, which can be found by detecting corresponding key points between sub-images. We upload these images on `Imagedata->Calibration->Homography->*`. Then we align all sub-images to a reference sub-image through perspective warping using the calibrated homographies. The calibration targets are under `Imagedata->Calibration->Texture->texture1->*`, where there are 10 folders showing different textures captured 100 times. For simplicity we just upload one texture. Note that a bright but unsaturated sub-image will lead to better accuracy, so we set different exposure time for each texture. The unnormalized power ratio $\alpha_i$ is $$\alpha_i = \frac{t_2}{t_i}\cdot\frac{\sum_{x,y} \|\nabla I_i(x,y)\|^2 }{\sum_{x,y} \nabla I_i(x,y) \cdot \nabla I_2(x,y)}$$
which is the least square solution between the $i_{th}$ and $2nd$ sub-image in the gradient domain

Run 
```
python calibration.py
```
for details.

We used a gradient-based fusion algorithm to effectively eliminate the optical artifact introduced by the imperfectness of the metasurface. We establish 5 HDR scenes and compare the visual quality and cross section of images captured from convention cameras and MetaHDR. The result demonstrates our system increases the dynamic range while preserving the LDR information.

Run
```
python HDRreconstruction.py
```
for reference

## Sample Result - Metasurface Simulation and Quantitative Analysis
![1simulated-psf](https://github.com/guo-research-group/MetaHDR/assets/149278360/b3d081ad-9d83-46e5-bc8f-2b9802e58381)

## Sample Result - HDR Reconstruction
![HDRresult](https://github.com/guo-research-group/MetaHDR/assets/149278360/c1f71577-c50d-4244-bd54-44890925c64b)

## Sample Result - HDR Video
![HDRvideo_flame](https://github.com/guo-research-group/MetaHDR/assets/149278360/1f8b50e5-17e8-4ec4-984c-9f062150709a)
![HDRvideo_circuit board](https://github.com/guo-research-group/MetaHDR/assets/149278360/89707636-ac4c-49ab-838b-049d5df8a25d)


## Citation
```
@article{MetaHDR:24,
author = {Charles Brookshire and Yuxuan Liu and Yuanrui Chen and Wei Ting Chen and Qi Guo},
journal = {Opt. Express},
keywords = {Chemical vapor deposition; Computational imaging; Electron beam lithography; Image metrics; Imaging systems; Scanning electron microscopy},
number = {15},
pages = {26690--26707},
publisher = {Optica Publishing Group},
title = {MetaHDR: single shot high-dynamic range imaging and sensing using a multifunctional metasurface},
volume = {32},
month = {Jul},
year = {2024},
url = {https://opg.optica.org/oe/abstract.cfm?URI=oe-32-15-26690},
doi = {10.1364/OE.528270},
}
```
