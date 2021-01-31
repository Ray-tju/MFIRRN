# MFIRRN
## Dense aligment Results
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image02428_ll_3DDFA.jpg" width="300" height="300" alt=""/><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image04292_ll_3DDFA.jpg" width="300" height="300" alt=""/>

## 3D Face Reconstruction Result
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image02428_ll_3DDFA.jpg" width="200" height="200" alt=""/><span>->->-></span><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/man_mesh_notexture.jpg" width="150" height="200" alt=""/><span>->->-></span><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/man_mesh.jpg" width="150" height="200" alt=""/>

## Quantitative Results
 NME2D   | AFLW2000-3D Dataset (68 pts)  | AFLW Dataset (21 pts)
:-: | :-: | :-: 
Method |[0,30],[30,60],[60,90], Mean, Std  | [0,30],[30,60],[60,90], Mean, Std
CDM | -, -, -, -, - | 8.150, 13.020, 16.170, 12.440, 4.040 
RCPR | 4.260, 5.960, 13.180, 7.800, 4.740 | 5.430, 6.580, 11.530, 7.850, 3.240
ESR | 4.600, 6.700, 12.670, 7.990, 4.190 | 5.660, 7.120, 11.940, 8.240, 3.290
SDM | 3.670, 4.940, 9.760, 6.120, 3.210 | 4.750, 5.550, 9.340, 6.550, 2.450 
DEFA  | 4.500, 5.560, 7.330, 5.803, 1.169 | -, -, -, -, - 
3DDFA(CVPR2016)  | 3.780, 4.540, 7.930, 5.420, 2.210 | 5.000, 5.060, 6.740, 5.600, 0.990
Nonlinear(CVPR2018)   | -, -, -, 4.700, - | -, -, -, -, -
DAMDNet(ICCVW19)  | 2.907, 3.830, 4.953, 3.897, 0.837 | 4.359, 5.209, 6.028, 5.199, 0.682 
MFIRRN  | 2.841, 3.572, 4.561, 3.658, 0.705 | 4.321, 5.051, 5.958, 5.110, 0.670 

## Qualitative Results of Dense Aligment
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/Dense.jpg" width="500" height="500">

## Qualitative Results of 3D Reconstruction 
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/3d.jpg" width="500" height="500">

## Installation
First you have to make sure that you have all dependencies in place.

You can create an anaconda environment called `mfirrn` using
```
conda env create -n mfirrn python=3.6 ## recommended python=3.6+
conda activate mfirrn
sudo pip3 install torch torchvision 
sudo pip3 install numpy scipy matplotlib
sudo pip3 install dlib
sudo pip3 install opencv-python
sudo pip3 install cython
```
Then, download the baseline code.
* download the [3DDFA](https://github.com/cleardusk/3DDFA)

Next, compile the extension modules.
```
cd utils/cython
python3 setup.py build_ext -i
```
Final, adopt our model in baseline code.
```
Copy our model to baseline
Replace train.py with our train.py
```

## Generation
To generate results using a trained model, use
```
python3 main.py -f samples/test.jpg 
```
Note that we suggest you choose normal image due to dlib restrictions on face capture

## Training
To train our MFIRRN with wpdc Loss, use
```
cd training
bash train_wqdc.sh
```

