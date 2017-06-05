# Image-Classification-CNN

1) Delete all the checkpoints from the "model" directory before training a new model from scratch.

2) Run the following command. This will read all the images from the dataset folder and split them into training and testing set and pickle them. This is done to avoid loading the images multiple times, so skip this step if you have already done this before.
```python
python prepareDataSetFromImages.py 
```

3) Run the following command to train the models. 
```python
python convNetTrain.py
```

The file <b> config.py </b> contains the parameters/flags.
