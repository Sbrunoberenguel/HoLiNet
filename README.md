# Visual Gyroscope
## Step 1: HoLiNet

In our article *Visual Gyroscope: Combination of Deep Learning Features and Direct Alignment for Panoramic Stabilization*, we propose a 3-step pipeline for panoramic image stabilization. In this repository can be found the first step of the pipeline, the Convolutional Neural Network **HoLiNet**

## Steps 2 and 3: MPP and PVG

The second and third steps of the pipeline share a common library that can be found in [LibPeR](https://github.com/PerceptionRobotique/libPeR_base)

Comming soon: [Visual Gyroscope](https://github.com/PerceptionRobotique/VisualGyroscope) 

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
