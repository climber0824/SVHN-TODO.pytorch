### (HOMEWORK)

# SVHN-TODO.pytorch


## Requirements

* Python 3.6
* PyTorch 0.4.1


## Setup

1. Download [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/) format 1

1. Extract to data folder, now your folder structure should be like below:
    ```
    SVHN-TODO.pytorch
        - data
            - extra
                - 1.png
                - 2.png
                - ...
                - digitStruct.mat
            - test
                - 1.png
                - 2.png
                - ...
                - digitStruct.mat
            - train
                - 1.png
                - 2.png
                - ...
                - digitStruct.mat
    ```


## Usage

1. Train
    ```
    $ python train.py -d=./data -c=./checkpoints
    ```

1. Evaluate
    ```
    $ python eval.py ./checkpoint/model-100.pth -d=./data
    ```

1. Infer
    ```
    $ python infer.py ./images/15.png -c=./checkpoints/model-100.pth
    $ python infer.py ./images/62.png -c=./checkpoints/model-100.pth
    ```


## References

* [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](http://arxiv.org/pdf/1312.6082.pdf)
* [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)

## Appendix

Numbers of images for each length, not including extra dataset.  

| length  |  train  |  test
|---------|---------|--------
|    0    |    0    |    0
|    1    |   5137  |   2483
|    2    |  18130  |   8356
|    3    |   8691  |   2081
|    4    |   1434  |   146
|    5    |    9    |    2
|    6    |    1    |    0
|  total  |  33402  |  13068
