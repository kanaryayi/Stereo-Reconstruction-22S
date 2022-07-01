# Stereo Reconstruction 22S

3D Scanning and Motion Capture Final Project

## Dataset 
We use [Middlebury 2014 Dataset](https://vision.middlebury.edu/stereo/data/scenes2014/). You can find some images for testing in directory `./data/Middlebury_2014/`. We're also open to include some different dataset (maybe outdoor scenes) for performance comparing later.

Here is a sample of Middlebury Dataset.

|                           Left                           |                           Right                           |
| :------------------------------------------------------: | :-------------------------------------------------------: |
| ![Left](data/Middlebury_2014/Adirondack-perfect/im0.png) | ![Right](data/Middlebury_2014/Adirondack-perfect/im1.png) |

## Getting started

I recommend to use vcpkg to manage our libraries, which is a package manager tool just like the pip for Python, maven for Java or npm for JavaScript. It's really convenient to use it to make our lives easier from building a bunch of related dependencies and so on. 

At first, you need to install vcpkg. You can find the installation tutorial [here](https://vcpkg.io/en/getting-started.html).

And then you'll need to install our dependencies.

For now, we only use Eigen and OpenCV.

- `vcpkg install eigen3:x64-windows`
- `vcpkg install opencv[contrib,nonfree]:x64-windows` (contrib, nonfree tag for using `xfeatures2d`)

`for MacOs you need to follow this issue to build opencv with opencv_contrib to have xfeatures2d https://github.com/udacity/SFND_2D_Feature_Tracking/issues/3`


After installing these two libraries, you should be able to build the project by using Cmake.

**Before Building:** Dont forget to change the path to your vcpkg in [`CmakeLists.txt`](CMakeLists.txt) at ***Line 4***.

## Project structure
The project folder should look like as following.
```
.
├── Data
│   └── Middlebury_2014
│       └── ...
├── Build
├── CMakeLists.txt
├── main.cpp
├── SURF.cpp
├── SURF.h
├── ...
└── README.md
```
## Members
- Yigit Burdurlu (03722506)
- Sebastian Bauer (03764577)
- Chang Luo (03759570)
