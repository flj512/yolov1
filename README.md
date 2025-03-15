# Train
This repo use pytorch to train the yolov1 models.  
The code is usefull for understanding the origin paper as well as test how to train a model    
and how the hyper parameters affect the traing process.  

## Python Env
Tested Python Verson: 3.10.12  
Torch Version: 2.5.1     
Other dependencies: OpenCV, Numpy, PIL  

## Download VOC2012 dataset
the following jupter will download the dataset and extra automatically.  

```
voc_download.ipynb
```

## Training
change values in config.py to do experiments
```
python3 train.py
```
# Validation
You can inference on one picture or multiple pictures    
one picture    
```
python3 inference.py
```
Example:    
![Airplane](https://github.com/flj512/yolov1/blob/master/output.jpg)

visulize inference result in jutper
```
vis_validation.ipynb
```
