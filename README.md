# Visual Gyroscope
## Step 1: HoLiNet

In our article [Visual Gyroscope: Combination of Deep Learning Features and Direct Alignment for Panoramic Stabilization](https://openaccess.thecvf.com/content/CVPR2023W/OmniCV/papers/Berenguel-Baeta_Visual_Gyroscope_Combination_of_Deep_Learning_Features_and_Direct_Alignment_CVPRW_2023_paper.pdf), we propose a 3-step pipeline for panoramic image stabilization. In this repository can be found the first step of the pipeline, the Convolutional Neural Network **HoLiNet**.

This one allows a first estimation of the Roll and Pitch angles with a global $360^{\circ}$ convergence domain.

## Steps 2 and 3: MPP and PVG

The second step of the pipeline uses the **Mixture of Photometric Potentials** (MPP) and performs an estimation of the remaining angle (*i.e.* Yaw). Like **Holinet**, this method can converge with a $360^{\circ}$ domain.

To refine the orientation measure, the last step of the pipeline relies on the sole **Photometric** information (PVG) to allow a Roll-Pitch-Yaw orientation estimation leading to a remaining estimated error of a few degrees. 

MPP and PVG share the same library that can be found in [LibPeR](https://github.com/PerceptionRobotique/libPeR_base)

The usage of the library for the visual gyroscope is detailled in the example routines available in the following repository: [Visual Gyroscope](https://github.com/PerceptionRobotique/VisualGyroscope) 

# HoLiNet usage

In order to use HoLiNet, we recomend to use [Anaconda](https://anaconda.org) virtual environments in Python.

First, we set up a virtual environment and activate it as:
```bash
conda create --name Gyro --file Gyro.txt

conda activate Gyro
```

An example of use of our network is:
```bash
python Gyroscope.py --root_dir path/to/images/to/compensate --pth path/to/network/weights  
```

The weights used in our article are available [here](https://drive.google.com/drive/folders/1hhXkx2x0dEZbxGYl1Mr3rwGWDuEYodmu?usp=sharing).

# Note from the authors
This code has not been thoroughly tested, which means it may have some bugs. **Please use with caution.**

The authors and developers of this code want the best for the users and have gone to great lengths to make it easy to use and accessible. 
Be nice to them and enjoy their work.

If any problem may appear, do not hesitate and we will do our best to solve it (at least we will try).

# License
This software is under GNU General Public License Version 3 (GPLv3), please see GNU License

If you have any problem, please contact the authors: Bruno Berenguel-Baeta (berenguel@unizar.es), Antoine N. Andre (antoine.andre@cnrs.fr), Guillaume Caron (guillaume.caron@u-picardie.fr), Jesús Bermudez-Cameo (bermudez@unizar.es) and Josechu Guerrero (josechu.guerrero@unizar.es)

You can also find extra information of this (and other) code in our respective GitHub accounts:
- [Bruno Berenguel-Baeta](https://github.com/Sbrunoberenguel)
- [Antoine N. Andre](https://github.com/AntoineAndre)
- [Guillaume Caron](https://github.com/GuicarMIS)
- [Jesus Bermudez](https://github.com/jesusbermudezcameo)
