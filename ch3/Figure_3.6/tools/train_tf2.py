# -*- coding:utf-8 -*-
# @Time  : 2022/3/13 8:36
# @Author: STARain
# @File  : train_tf2.py

#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

#save 模型/网络中的各训练参数
def save_trainable_vars(model,filename,**kwargs):
    """save a .npz archive in `filename`  with
    the current value of each variable in tf.trainable_variables()
    plus any keyword numpy arrays.
    """
    save={}
    for v in model.trainable_variables:
        save[str(v.name)] = v.numpy()
    save.update(kwargs)
    np.savez(filename,**save)


def load_trainable_vars(model,filename):
    """load a .npz archive and assign the value of each loaded
    ndarray to the trainable variable whose name matches the
    archive key.  Any elements in the archive that do not have
    a corresponding trainable variable will be returned in a dict.
    """
    other={}
    try:
        tv=dict([ (str(v.name),v) for v in model.trainable_variables ])
        for k,d in np.load(filename).items():
            if k in tv:
                print('restoring ' + k)
                tv[k].assign(tf.convert_to_tensor(d))
            else:
                other[k] = d
    except IOError:
        pass
    return other

