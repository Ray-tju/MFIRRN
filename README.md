# MFIRRN
## Results
<img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image02428_ll_3DDFA.jpg" width="500" height="500" alt=""/><img src="https://github.com/leilimaster/MFIRRN/blob/main/display/image04292_ll_3DDFA.jpg" width="500" height="500" alt=""/>

## Installation
First you have to make sure that you have all dependencies in place.

You can create an anaconda environment called `dmifnet_space` using
```
conda env create -n mfirrn python=3.6 ## recommended python=3.6+
conda activate mfirrn
sudo pip3 install torch torchvision 
sudo pip3 install numpy scipy matplotlib
sudo pip3 install dlib
sudo pip3 install opencv-python
sudo pip3 install cython
```

Then, compile the extension modules.
```
cd utils/cython
python3 setup.py build_ext -i
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
