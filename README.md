# Clothe-right

## Installing Dependencies 

```
pip install -r requirements.txt
git submodule update --init
cd pymeanshift
./setup.py install
```

## Download dataset and prepare data

```
./init.sh
```

Running this will download the [clothing-co-parsing](https://github.com/bearpaw/clothing-co-parsing) and then prepare the data and store it in the ```clean_data``` directory, as well as create the ```checkpoints``` directory.

## Training the model

```
./train.py
```

This will save the models with the best accuracies under the ```checkpoints```
directory.

## Detecting clothes

To pass the whole image to the CNN and find the segments which are shadows:
```
./detect.py --image <path to image> --model <path to model> --size <size>
```


```<path to image>``` is the path to any image.

```<path to model>``` is the to either ```model.h5``` or any of the models under
the ```checkpoints``` directory.

```<size>``` is the input dimensions of the model ```sizexsize```.
