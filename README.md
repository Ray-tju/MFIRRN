# MFIRRN
## Dense aligment Results
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image02428_ll_3DDFA.jpg" width="300" height="300" alt=""/><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image04292_ll_3DDFA.jpg" width="300" height="300" alt=""/>

## 3D Face Reconstruction Result
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image02428_ll_3DDFA.jpg" width="200" height="200" alt=""/><span>->->-></span><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/man_mesh_notexture.jpg" width="150" height="200" alt=""/><span>->->-></span><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/man_mesh.jpg" width="150" height="200" alt=""/>

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

##Training
To train our MFIRRN with wpdc Loss, use
```
cd training
bash train_wqdc.sh
```
