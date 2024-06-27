# MetaHDR: Single Shot High-dynamic Range Imaging and Sensing Using a Multifunctional Metasurface

_Charles Brookshire, Yuxuan Liu, Yuanrui Chen, and Qi Guo_

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: liu3910@purdue.edu


[Paper](https://preprints.opticaopen.org/articles/preprint/MetaHDR_Single_Shot_High-Dynamic_Range_Imaging_and_Sensing_using_a_Multifunctional_Metasurface/25719513) | [Visualization 1](https://drive.google.com/file/d/1xR25B3CRW_7aeHa4g03yvT2yfVWkDwZg/view?usp=drive_link) | [Visualization 2](https://drive.google.com/file/d/18klBNhE-oL3ldxwv8qmt05sb1dHNUDW9/view?usp=drive_link)
## Introduction

MetaHDR is a single-shot high-dynamic range (HDR) imaging and sensing system using a multifunctional metasurface. The system can simultaneously form 9 low dynamic range (LDR) images with distinct irradiance on a photosensor, and fuse them into HDR image using a gradient-based algorithm. MetaHDR achieves single-shot HDR photography and videography that increases the dynamic range by at least 50 dB compared to the original dynamic range of the photosensor. 

## Usage
If you want to simulate the lens surface and profile, please make sure you have successfully installed DFlat, which can be done via
```
pip install dflat_opt
```
or
```
git clone https://github.com/DeanHazineh/DFlat
pip install -e .
```
Then you can use our code, e.g., reconstruct a HDR image, by
```
git clone https://github.com/guo-research-group/MetaHDR
cd MetaHDR
python HDRreconstruction.py
```
## Sample Result - Metasurface Simulation
![github_result2](https://github.com/guo-research-group/MetaHDR/assets/149278360/ac4aee93-6d48-45ac-9ba1-4b9aeed5389d)

## Sample Result - HDR reconstruction
![github_result1](https://github.com/guo-research-group/MetaHDR/assets/149278360/85f47837-7f6f-408d-b3a2-17d0c4d84d5e)
