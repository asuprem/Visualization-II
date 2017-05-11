import os, pdb, json, re, keras, vis_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, load_model, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, InputLayer
from keras.utils import np_utils, generic_utils
from keras import backend as K
from PIL import Image
import numpy as np
from tqdm import tqdm
import gc


model_json = vis_utils.json_load('models.json')


model_type = 'cifar'
top_level= [_dir for _dir in os.listdir(model_type) if not _dir.startswith('.')]
dir_paths = {entry:os.path.join(model_type, entry) for entry in top_level}
jsons = os.listdir(dir_paths['jsons'])
model_dirs = [os.path.splitext(_item)[0] for _item in jsons]



_BASEDIM = model_json[model_type]['vis_dim'][0]
vis_root = np.random.random((1,_BASEDIM, _BASEDIM,3))*20+110
for idx in tqdm(range(len(jsons)), desc="Models"):
    if jsons[idx]=='cifar_6.json':
	continue
    print('Starting with ' + jsons[idx])
    #idx = 0
    dir_list = vis_utils.dir_list_returner(dir_paths, ['weights','max_activations','history','supplement'], model_dirs[idx])
    #Set up supplemental parameters
    compile_params = vis_utils.json_load(os.path.join(dir_list['supplement'], 'compile.json'))
    activation_params = vis_utils.json_load(os.path.join(dir_list['supplement'], 'activation.json'))
    activation_layers = [str(_layer) for _layer in activation_params['snip_layer']]
    # Loading the Model file
    with open(os.path.join(dir_paths['jsons'],jsons[idx]), 'r') as model_file:
        model = model_from_json(model_file.read())

    weight_files = [_file for _file in os.listdir(dir_list['weights']) if _file.startswith('weights')]

    for wt_idx in tqdm(range(len(weight_files)), desc = jsons[idx], leave=False):
    #wt_idx = 5
        weight_folder = os.path.join(dir_list['max_activations'], 'epoch'+str(wt_idx))
        make = os.makedirs(weight_folder) if not os.path.exists(weight_folder) else None
        # Building the second model and set the weights
        model.load_weights(os.path.join(dir_list['weights'], weight_files[wt_idx]))
        vis_model = vis_utils.snip_build(model, activation_layers)
        #extract the convolutional layers
        conv_layers = [str(_layer.name) for _layer in vis_model.layers if _layer.name.startswith('conv')]
        #layerDict={}
        #for _name in conv_layers:
        #        layerDict[_name] = {'name':_name}
        #        layerDict[_name]['kernels'] = vis_model.get_layer(_name).get_config()['filters']
        #        layerDict[_name]['sources'] = []
        #vis_layer = conv_layers[-1]
        for lyr_idx, vis_layer in enumerate(tqdm(conv_layers, desc="Layer", leave=False)):
        #for lyr_idx, vis_layer in enumerate(conv_layers):
            conv_folder = os.path.join(weight_folder, vis_layer)
            make = os.makedirs(conv_folder) if not os.path.exists(conv_folder) else None
            #_BASEDIM = model_json[model_type]['vis_dim'][0]
            kernel_size = vis_model.get_layer(vis_layer).get_config()['filters']
            for knl_idx, vis_kernel in enumerate(tqdm(range(kernel_size), desc="Kernel", leave=False)):
            #for knl_idx, vis_kernel in enumerate(range(kernel_size)):
                kernel_file = os.path.join(conv_folder, 'kernel' + str(vis_kernel)+'.png')
                # Create random root image
                vis_iters = model_json[model_type]["vis_iters"]
                image = vis_utils.visualize_filter(vis_kernel, vis_root, vis_layer, vis_model, vis_iters)
                #layerDict[vis_layer]['sources'].append(vis_utils.visualize_filter(vis_kernel, vis_root, vis_layer, vis_model, vis_iters))
                Image.fromarray(vis_utils.tensor_to_image(image)).save(kernel_file)
                #gc.collect()
