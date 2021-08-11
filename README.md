PyTorch implementation based on the paper "[Invertible Autoencoder for Domain Adaptation](https://www.mdpi.com/2079-3197/7/2/20/htm)"

Network for the 127x127 input size images.

In order to run: 
1. Download Road Data from [KAIST](https://soonminhwang.github.io/rgbt-ped-detection/ "Download speeds are horrible, look for the OneDrive Link") Dataset (set01.zip and set04.zip), 
2. Extract and put images from the visible subfolders to  ```./data/day/``` and ```./data/night/``` folders respectively
3. Run ```train.py ````

Code is written in Python 3.8 with PyTorch 1.8.0
