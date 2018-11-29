import os
import h5py
path_to_mat = os.path.join('./data/train/digitStruct.mat')
_h5py_file = h5py.File(path_to_mat)
_h5py_data = _h5py_file.get('digitStruct')
_name = _h5py_data.get('name')
print(len(_name))
_name_ref = _name[33402][0]
_obj_name = _h5py_data.get(_name_ref)
_image_filename = ''.join(chr(i) for i in _obj_name[:])
print(_image_filename)