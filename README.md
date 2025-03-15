# Train
This repo use pytorch to train the yolov1 model.  
The code is usefull for understanding the origin paper as well as test how to train a model    
and how the hyper parameters affect the traing process.  

## Python Env
Tested Python Verson: 3.10.12  
Envirentment: [requirements.txt](https://github.com/flj512/yolov1/blob/master/requirements.txt)

## Download VOC2012 dataset
the following command download the dataset and extra automatically.  

```
python3 voc_download.py
```

## Training
change values in config.py to do experiments
```
python3 train.py
```
it may success if the train loss and val loss decrease to around 4

![Loss](https://github.com/flj512/yolov1/blob/master/loss.png)

# Validation
You can inference on one picture or multiple pictures    

One picture    
```
python3 inference.py
```
The result will write to output.jpg. 
  
Example:    
![Airplane](https://github.com/flj512/yolov1/blob/master/output.jpg)

OR visualize inference results in jupyter
```
vis_validation.ipynb
```
