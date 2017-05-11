import re, json, os, keras, math
import numpy as np
from cv2 import blur
from keras.models import Sequential
from keras.layers import InputLayer
from keras import backend as K
import pdb
from tqdm import tqdm, tqdm_notebook, tnrange


def json_load(filepath):
    _file = open(filepath,'r')
    _file_params = _file.read()
    _file.close()
    _file_params = json.loads(_file_params)
    return _file_params


def snip_test(model, activation_layers):
    vis_model = Sequential()
    vis_model.add(InputLayer((None,None, 3)))
    #Necessary cause of a stupid keras bug
    layerCount = 1
    for layer in model.layers:
        if layer.name == 'i_layer':
            continue
        nLayer = keras.layers.deserialize({'class_name': layer.__class__.__name__,
                                'config': layer.get_config()})
        if layer.name not in activation_layers:
            vis_model.add(nLayer)
            vis_model.layers[layerCount].trainable = False
        else:
            break
    return vis_model

def snip_build(model, activation_layers):
    vis_model = Sequential()
    vis_model.add(InputLayer((None,None, 3)))
    #Necessary cause of a stupid keras bug
    layerCount = 1
    for layer in model.layers:
        #if re.match('([a-z]+_)', layer.name).group(0)[:-1] == 'input':
        if layer.name == 'i_layer':
            continue
        nLayer = keras.layers.deserialize({'class_name': layer.__class__.__name__,
                                'config': layer.get_config()})
        if layer.name not in activation_layers:
            #print nLayer.trainable, nLayer.name
            #nLayer.set_weights(layer.get_weights())
            #pdb.set_trace()
            vis_model.add(nLayer)
            vis_model.layers[layerCount].set_weights(layer.get_weights())
            vis_model.layers[layerCount].trainable = False
            layerCount+=1
            #vis_model.layers[layerCount].trainable = False
            #if layer.name not in activation_layers else '__snip__'
        else:
            break
        #if model_add == '__snip__':
        #    break
        #print nLayer.name
        #nLayer.set_weights(layer.get_weights())
        
    return vis_model

# This returns the model directory for each top-level folder, i.e. cifar/history/cifar_6 for cifar_6's history folder
def dir_list_returner(dir_paths, dir_list, _file):
    dir_list = [dir_paths[_dir] for _dir in dir_list]
    dir_list = [os.path.join(_dir, _file) for _dir in dir_list]
    dir_list = {re.search('(?<=/).[a-z_]+',_dir).group(0):_dir for _dir in dir_list}
    return dir_list


def learning_rate(nu, _iters, func="const"):
    _iters+=1
    if func == 'square_root':
        return nu/math.sqrt(_iters)
    if func == 'const':
        return nu
    if func == 'linear':
        return nu/_iters
    if func == 'square':
        return (nu*nu)/_iters
    return nu

def cvblur(image, blur_size=(3,3)):
    return np.array([blur(image[0], blur_size)])

def decay(image, decay_param = 0.9):
    return image*decay_param

def pixel_clip(image, pct = [1,None]):
    image[np.where(np.abs(image) < np.percentile(np.abs(image),pct[0]))]=0
    if pct[1] is not None:
        assert(pct[1]>pct[0])
        image[np.where(np.abs(image) > np.percentile(np.abs(image),pct[1]))]=0
    return image

def vis_parse(vis_params):
    if vis_params is not None:
        vis_params['reg_weights'] = [float(_i)/sum(vis_params['reg_weights']) for _i in vis_params['reg_weights']]
        return vis_params['reg_params'],vis_params['reg_weights']
    reg_params = [(3,3),.9,[1,None]]
    reg_weights = [3,3,1]
    reg_weights = [float(_i)/sum(reg_weights) for _i in reg_weights]
    return reg_params, reg_weights

#vis_params: dict object with two entries: 'reg_params' and 'reg_weights'
# reg_params: parameters for regularizations in the order as follows:
    #cvblur -> tuple with size info: (3,3), or (4,4)
    #decay  -> numerical parameter: 0.9, 0.8
    #pixelclip -> clipping percentages = 1 and upper percentile; upper not necessary.
    #median -> aperture -> 5
    #should be a list, i.e. [(3,3),0.9, [1,None], 5]
# reg_weights is a list of weight of the parametrization. Standard weights are [3,2,1] for blur, decay and clip, respectively
# Several samples for vis_params are available in vis_params.json
        
def visualize_filter(kernel_index, root_image, model_layer, model, iterations=20, alpha=0.9, vis_params = None):
    reg_params, reg_weights = vis_parse(vis_params)
    in_image = model.input
    #pdb.set_trace()
    model_layer = model.get_layer(model_layer).output
    loss = K.mean(model_layer[:,:,:,kernel_index])
    gradients = K.gradients(loss,in_image)[0]
    #L2 normalization
    gradients /= K.sqrt(K.mean(K.square(gradients))) + 1e-6
    lossGrad = K.function([in_image, K.learning_phase()], [loss, gradients])
    #lossGrad = K.function([in_image, K.learning_phase()], [gradients])

    for _iters in range(iterations):
        #lossN, gradN = lossGrad([root_image,1])
        lossN, gradN = lossGrad([root_image,0])
        #pdb.set_trace()
        root_image += gradN*learning_rate(alpha, _iters, 'const')
        regularizers = [cvblur, decay, pixel_clip]
        regularized = [function(root_image, param) for function, param in zip(regularizers, reg_params)]
        root_image = np.array([_weight*_image for _weight, _image in zip(reg_weights, regularized)])
        root_image = np.sum(root_image, axis=0)
    return root_image

def get_layers_coded(vis_model, code):
    return [str(_layer.name) for _layer in vis_model.layers if _layer.name.startswith(code)]

def get_conv_layers(vis_model):
    return get_layers_coded(vis_model, 'conv')

def build_cg(vis_model):
    vis_conv = get_conv_layers(vis_model)
    vis_cg = {layers:{'name':layers} for layers in vis_conv}
    
    for layer in vis_conv:
        current_layer = vis_model.get_layer(layer).output    
        model_layer = vis_model.get_layer(layer).output

        loss = K.mean(model_layer, axis=(0,1,2))
        grads=[]
        for i in tqdm_notebook(range(loss.get_shape().as_list()[0]), desc="Grads compute",leave=False):
                grads.append(K.gradients(loss[i],vis_model.input)[0])
        ngrad=[]
        for g_idx, _grad in enumerate(tqdm_notebook(grads, desc="Grads update", leave=False)):
                ngrad.append(_grad/(K.sqrt(K.mean(K.square(_grad))) + 1e-6))
        #lossGrad
        vis_cg[layer]['gradients'] = [K.function([vis_model.input, K.learning_phase()], [loss[i], ngrad[i]]) for i in range(len(ngrad))]
    return vis_cg


    
def tensor_to_image(tensor):
    return (np.clip((0.1*((tensor - tensor.mean())/(tensor.std() + 1e-6)))+.5,0,1)[0]*255).astype("uint8")




def visualize_filter_cg(kernel_index, root_image, model_layer, compute_graph, iterations=20, alpha=0.9, vis_params = None):
    reg_params, reg_weights = vis_parse(vis_params)
    for _iters in range(iterations):
        #lossN, gradN = lossGrad([root_image,1])
        lossN, gradN = compute_graph[model_layer]['gradients'][kernel_index]([root_image,0])
        #pdb.set_trace()
        root_image += gradN*learning_rate(alpha, _iters, 'const')
        regularizers = [cvblur, decay, pixel_clip]
        regularized = [function(root_image, param) for function, param in zip(regularizers, reg_params)]
        root_image = np.array([_weight*_image for _weight, _image in zip(reg_weights, regularized)])
        root_image = np.sum(root_image, axis=0)
    return root_image

#def activations()