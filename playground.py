import os
import h5py
import numpy as np

# path_to_mat = os.path.join('./data/train/digitStruct.mat')
# _h5py_file = h5py.File(path_to_mat)
# _h5py_data = _h5py_file.get('digitStruct')
# _name = _h5py_data.get('name')
#
# # print(len(_name))
# map_of_bbox = {}
# lengthes = []
# for index in range(len(_name)):
#     item = _h5py_file['digitStruct']['bbox'][index].item()
#     for key in ['label', 'left', 'top', 'width', 'height']:
#         attr = _h5py_file[item][key]
#         values = [_h5py_file[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [
#             attr.value[0][0]]
#         map_of_bbox[key] = values
#
#     length = len(map_of_bbox['label'])
#     lengthes.append(length)

    # if length == 0:
    #     _name_ref = _name[index][0]
    #     _obj_name = _h5py_data.get(_name_ref)
    #     _image_filename = ''.join(chr(i) for i in _obj_name[:])
    #     print(_image_filename)

# print(np.unique(lengthes))


# path_to_mat = os.path.join('./data', 'train', 'digitStruct.mat')
# _h5py_file = h5py.File(path_to_mat)
# _h5py_data = _h5py_file.get('digitStruct')
# _name = _h5py_data.get('name')
# for index in range(len())
# _name_ref = _name[index][0]
# _obj_name = _h5py_data.get(_name_ref)
# _image_filename = ''.join(chr(i) for i in _obj_name[:])
# _path_to_image = os.path.join(self._path_to_data, self._mode.value, _image_filename)
# image = Image.open(_path_to_image)