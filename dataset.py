from enum import Enum
from typing import Tuple

import PIL
import torch.utils.data
import glob
import os
import h5py

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from PIL import Image
from torch import Tensor
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    class Mode(Enum):
        TRAIN = 'train'
        TEST = 'test'

    def __init__(self, path_to_data_dir: str, mode: Mode):
        super().__init__()
        is_train = mode == Dataset.Mode.TRAIN

        # TODO: CODE BEGIN
        # raise NotImplementedError
        self._path_to_data = path_to_data_dir
        self._mode = mode
        self._length = len(glob.glob(os.path.join(self._path_to_data, self._mode.value, '*')))
        if is_train:
            self._length += len(glob.glob(os.path.join(self._path_to_data, 'extra/*')))
        # TODO: CODE END

    def __len__(self) -> int:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        return self._length
        # TODO: CODE END

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        # image_path = sorted(glob.glob(os.path.join(self._path_to_data, self._mode.value, '*')))[index]
        # image = Image.open(image_path)
        # image = np.array(image)
        # image = Image.fromarray(image)
        # image = self.preprocess(image)

        path_to_mat = os.path.join(self._path_to_data, self._mode.value, 'digitStruct.mat')
        _h5py_file = h5py.File(path_to_mat)
        _h5py_data = _h5py_file.get('digitStruct')
        _bbox = _h5py_data.get('bbox')
        _name = _h5py_data.get('name')
        _name_ref = _name[index][0]
        _obj_name = _h5py_data.get(_name_ref)
        _image_filename = ''.join(chr(i) for i in _obj_name[:])
        print(_image_filename)
        _path_to_image = os.path.join(self._path_to_data, self._mode.value, _image_filename)
        image = Image.open(_path_to_image)
        image = self.preprocess(image)

        return image
        # TODO: CODE END

    @staticmethod
    def preprocess(image: PIL.Image.Image) -> Tensor:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        transform = transforms.Compose([
            # transforms.RandomCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        return transform(image)
        # TODO: CODE END


if __name__ == '__main__':
    _dataset = Dataset(path_to_data_dir='./data', mode=Dataset.Mode.TRAIN)
    a = _dataset[10][0]