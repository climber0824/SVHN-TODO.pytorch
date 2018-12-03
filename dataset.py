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
from sklearn.preprocessing import OneHotEncoder


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
        self._length = len(glob.glob(os.path.join(self._path_to_data, self._mode.value, '*.png')))
        # if is_train:
        #     self._length += len(glob.glob(os.path.join(self._path_to_data, 'extra/*')))
        # TODO: CODE END

    def __len__(self) -> int:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        return self._length
        # TODO: CODE END

    def __getitem__(self, index) -> Tuple[Tensor, Tensor, Tensor]:
        # TODO: CODE BEGIN
        # raise NotImplementedError
        # image_path = sorted(glob.glob(os.path.join(self._path_to_data, self._mode.value, '*')))[index]
        # image = Image.open(image_path)
        # print(type(image))
        # image = self.preprocess(image)
        path_to_mat = os.path.join(self._path_to_data, self._mode.value, 'digitStruct.mat')
        _h5py_file = h5py.File(path_to_mat)
        _h5py_data = _h5py_file.get('digitStruct')
        _name = _h5py_data.get('name')
        _name_ref = _name[index][0]
        _obj_name = _h5py_data.get(_name_ref)
        _image_filename = ''.join(chr(i) for i in _obj_name[:])
        _path_to_image = os.path.join(self._path_to_data, self._mode.value, _image_filename)
        image = Image.open(_path_to_image)
        image = image.resize((64, 64))
        image = self.preprocess(image)
        image = image.view(64, 64, 3)

        map_of_bbox = {}
        item = _h5py_file['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = _h5py_file[item][key]
            values = [_h5py_file[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [
                attr.value[0][0]]
            map_of_bbox[key] = values

        # Without One-Hot
        # length = len(map_of_bbox['label'])      # length = 1 to 5

        # With One-Hot
        a = len(map_of_bbox['label'])   # a = 1, 2, 3, 4, 5
        length = [0, 0, 0, 0, 0]
        length[a-1] = 1     # length = [0, 1, 0, 0, 0]

        # Without One-Hot
        # digits = [10, 10, 10, 10, 10]
        # for idx in range(length):
        #     digits[idx] = map_of_bbox['label'][idx]
        #     if digits[idx] == 10:
        #         digits[idx] = 0       # digits = [1, 9, 0, 0, 0, 0]

        # With One-Hot
        digits = []
        for idx in range(a):
            digit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            if map_of_bbox['label'][idx] == 10:
                pass
            else:
                digit[-1] = 0
                digit[int(map_of_bbox['label'][idx])] = 1
            digits.append(digit)
        digit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        for _ in range(5-a):
            digits.append(digit)

        length = torch.Tensor(length)
        digits = torch.Tensor(digits)
        return image, length, digits
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
    print(len(_dataset))
    _image, _length, _digits = _dataset[666]
    # print('length: %s' % _length.nonzero())
    # print('digits: %s' % _digits.nonzero())
    print('image type: %s' % type(_image))
    print('length type: %s' % type(_length))
    print('digits type: %s' % type(_digits))
