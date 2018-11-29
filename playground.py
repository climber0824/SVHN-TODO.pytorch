import os
import h5py
path_to_mat = os.path.join('./data/train/digitStruct.mat')
_h5py_file = h5py.File(path_to_mat)
_h5py_data = _h5py_file.get('digitStruct')
_name = _h5py_data.get('name')

map_of_bbox = {}
item = _h5py_file['digitStruct']['bbox'][9000].item()
for key in ['label', 'left', 'top', 'width', 'height']:
    attr = _h5py_file[item][key]
    values = [_h5py_file[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [
        attr.value[0][0]]
    map_of_bbox[key] = values

print(map_of_bbox['width'])