import h5py
import numpy as np

path_to_data = './data/train/digitStruct.mat'
file = h5py.File(path_to_data, 'r')     # h5py._hl.files.File
# for key in file.keys():
#     print(key)

data = file.get('digitStruct')      # h5py._hl.group.Group
_bbox = data.get('bbox')            # h5py._hl.dataset.Dataset
_name = data.get('name')            # h5py._hl.dataset.Dataset
ref_b = _bbox[15][0]                # h5py.h5r.Reference
obj_b = data.get(ref_b)             # h5py._hl.group.Group
_height = obj_b.get('height')       # h5py._hl.dataset.Dataset
_label = obj_b.get('label')         # h5py._hl.dataset.Dataset


n_label = np.array(_label)


ref_n = _name[15][0]                # h5py.h5r.Reference
obj_n = data.get(ref_n)             # h5py._hl.dataset.Dataset

# _filename = ''.join(chr(i) for i in obj_n[:])
# print(_filename)

