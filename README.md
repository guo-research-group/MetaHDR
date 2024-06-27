# MetaHDR: Single Shot High-dynamic Range Imaging and Sensing Using a Multifunctional Metasurface

_Charles Brookshire, Yuxuan Liu, Yuanrui Chen, and Qi Guo_

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: liu3910@purdue.edu


[Paper](https://preprints.opticaopen.org/articles/preprint/MetaHDR_Single_Shot_High-Dynamic_Range_Imaging_and_Sensing_using_a_Multifunctional_Metasurface/25719513) | [Visualization 1](https://drive.google.com/file/d/1xR25B3CRW_7aeHa4g03yvT2yfVWkDwZg/view?usp=drive_link) | [Visualization 2](https://drive.google.com/file/d/18klBNhE-oL3ldxwv8qmt05sb1dHNUDW9/view?usp=drive_link)
## Introduction

MetaHDR is a single-shot high-dynamic range (HDR) imaging and sensing system using a multifunctional metasurface. The system can simultaneously form 9 low dynamic range (LDR) images with distinct irradiance on a photosensor, and fuse them into HDR image using a gradient-based algorithm. MetaHDR achieves single-shot HDR photography and videography that increases the dynamic range by at least 50 dB compared to the original dynamic range of the photosensor. 

## Usage
To install our code, run
```
git clone https://github.com/guo-research-group/MetaHDR
cd MetaHDR
```

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

## Sample Result - Metasurface Simulation
![github_result2](https://github.com/guo-research-group/MetaHDR/assets/149278360/ac4aee93-6d48-45ac-9ba1-4b9aeed5389d)

## Sample Result - HDR reconstruction
![HDRresult](https://github.com/guo-research-group/MetaHDR/assets/149278360/c1f71577-c50d-4244-bd54-44890925c64b)



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
