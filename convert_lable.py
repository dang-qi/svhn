import numpy as np
import pickle
import os
import h5py
from PIL import Image
def convert_data(path, im_folder, id_prefix, obj_start_id):
    mat_f = h5py.File(path, 'r')
    names = get_names(mat_f)
    objs  = get_objs(mat_f)
    assert len(names) == len(objs)
    annos = []
    obj_id = obj_start_id
    
    for name,obj in zip(names,objs):
        im = {}
        im['file_name'] = name
        im['id'] = int('{}{:07d}'.format(id_prefix, int(name[:-4])))
        w,h=get_im_info(os.path.join(im_folder, name))
        im['height'] = h
        im['width'] = w
        for o in obj:
            o['image_id'] = int('{}{:07d}'.format(id_prefix, int(name[:-4])))
            o['id'] = int(obj_id)
            obj_id += 1
        
        im['objects'] = obj
        #print(im)
        annos.append(im)
    print('There are {} objects'.format(obj_id-obj_start_id-1))
    return annos

def get_im_info(im_path):
    im = Image.open(im_path)
    w,h = im.size
    return w, h


def get_objs(mat_f):
    box_obj = mat_f['digitStruct/bbox']
    objs = []
    for i in range(box_obj.size):
        objs_singel_image = get_box_single_image(i, box_obj, mat_f)
        objs.append(objs_singel_image)
    return objs

def bbox_helper(attr, mat_f):
    if len(attr) > 1:
        attr = [mat_f[attr[j].item()][0][0] for j in range(len(attr))]
    else:
        attr = [attr[0][0]]
    return attr

def get_box_single_image(n, box_obj, mat_f):
    objs = []
    bb = box_obj[n].item()
    # bbox = bboxHelper(f[bb]["label"])
    height = bbox_helper(mat_f[bb]["height"],mat_f)
    label = bbox_helper(mat_f[bb]["label"],mat_f)
    left = bbox_helper(mat_f[bb]["left"],mat_f)
    top = bbox_helper(mat_f[bb]["top"],mat_f)
    width = bbox_helper(mat_f[bb]["width"],mat_f)
    for x,y,w,h,l in zip(left, top, width, height, label):
        obj = {}
        obj['bbox'] = [x,y,w,h]
        obj['area'] = w*h
        obj['category_id'] = int(l)
        obj['iscrowd'] = 0
        objs.append(obj)
    return objs

def get_names(mat_f):
    names_obj=mat_f['digitStruct/name']
    names = []
    for i in range(names_obj.shape[0]):
        names.append(''.join([chr(v[0]) for v in mat_f[names_obj[i][0]]]))
    return names

if __name__=="__main__":
    dataset = ['train','test', 'extra']
    obj_start_ids = [0, int(1e7), int(2e7)]
    for i, d in enumerate(dataset):
        print('converting dataset part {}'.format(d))
        out_path = './data/{}.pkl'.format(d)
        data_path = './data/{}/digitStruct.mat'.format(d)
        im_folder = os.path.abspath('./data/{}').format(d)
        #print(im_folder)
        #mat_f = h5py.File(data_path,'r')
        #num_img = mat_f['/digitStruct/name'].size
        anno = convert_data(data_path, im_folder, id_prefix=i, obj_start_id=obj_start_ids[i])
        out_data = {}
        out_data[d] = anno
        with open(out_path, 'wb') as f:
            pickle.dump(out_data, f)