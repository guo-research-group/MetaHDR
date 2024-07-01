# MetaHDR: Single Shot High-dynamic Range Imaging and Sensing Using a Multifunctional Metasurface

_Charles Brookshire*, Yuxuan Liu*, Yuanrui Chen, Wei-Ting Chen, and Qi Guo_

*Co-first authors with equal contribution

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: liu3910@purdue.edu


[Paper](https://preprints.opticaopen.org/articles/preprint/MetaHDR_Single_Shot_High-Dynamic_Range_Imaging_and_Sensing_using_a_Multifunctional_Metasurface/25719513)
## Introduction

MetaHDR is a single-shot high-dynamic range (HDR) imaging and sensing system using a multifunctional metasurface. The system can simultaneously form 9 low dynamic range (LDR) images with distinct irradiance on a photosensor, and fuse them into HDR image using a gradient-based algorithm. MetaHDR achieves single-shot HDR photography and videography that increases the dynamic range by at least 50 dB compared to the original dynamic range of the photosensor. 

## Usage

### Installation
To install our code, run
```
git clone https://github.com/guo-research-group/MetaHDR
cd MetaHDR
```

### FDTD Simulation


### Metasurface Simulation
To simulate the lens surface and profile, please make sure you have successfully installed DFlat.
```
git clone https://github.com/DeanHazineh/DFlat
pip install -e .
python PSFsimulation.py
```

To reconstruct a HDR image, run
```
python HDRreconstruction.py
```

## Sample Result - Metasurface Simulation and Quantitative Analysis
![1simulated-psf](https://github.com/guo-research-group/MetaHDR/assets/149278360/8ebc5d39-7d2e-4397-b181-28f5ffff73ab)


## Sample Result - HDR Reconstruction
![HDRresult](https://github.com/guo-research-group/MetaHDR/assets/149278360/c1f71577-c50d-4244-bd54-44890925c64b)

## Sample Result - HDR Video
![HDRvideo_flame](https://github.com/guo-research-group/MetaHDR/assets/149278360/1f8b50e5-17e8-4ec4-984c-9f062150709a)
![HDRvideo_circuit board](https://github.com/guo-research-group/MetaHDR/assets/149278360/89707636-ac4c-49ab-838b-049d5df8a25d)


## Citation
```
@article{Brookshire2024,
author = "Charles Brookshire and Yuxuan Liu and Yuanrui Chen and Wei-Ting Chen and Qi Guo",
title = "{MetaHDR: Single Shot High-Dynamic Range Imaging and Sensing using a Multifunctional Metasurface}",
year = "2024",
month = "4",
url = "https://preprints.opticaopen.org/articles/preprint/MetaHDR_Single_Shot_High-Dynamic_Range_Imaging_and_Sensing_using_a_Multifunctional_Metasurface/25719513",
doi = "10.1364/opticaopen.25719513.v1"
}
```
