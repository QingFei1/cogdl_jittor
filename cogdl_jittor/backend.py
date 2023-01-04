#! /usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import sys
Default_BACKEND = 'jittor'
# BACKEND = 'torch'

# Check for backend.json files
cogdl_backend_dir = os.path.expanduser('~')
if not os.access(cogdl_backend_dir, os.W_OK):
    cogdl_backend_dir = '/tmp'
cogdl_dir = os.path.join(cogdl_backend_dir, '.cogdl')

config = {
    'backend': Default_BACKEND,
}
if not os.path.exists(cogdl_dir):
    path = os.path.join(cogdl_dir, 'cogdl_backend.json')
    os.makedirs(cogdl_dir)
    with open(path, "w") as f:
        json.dump(config, f)
    BACKEND = config['backend']
    sys.stderr.write("Create the backend configuration file :" + path + '\n')
else:
    path = os.path.join(cogdl_dir, 'cogdl_backend.json')
    with open(path, 'r') as load_f:
        load_dict = json.load(load_f)
        BACKEND = load_dict['backend']

# Set backend based on BACKEND.
if 'BACKEND' in os.environ:
    backend = os.environ['BACKEND']
    if backend in ['jittor', 'torch']:
        BACKEND = backend
    else:
        print("CogDL backend not selected or invalid.  "
                "Assuming PyTorch for now.")
        path = os.path.join(cogdl_dir, 'cogdl_backend.json')
        with open(path, "w") as f:
            json.dump(config, f)

# # import backend functions
# if BACKEND == 'tensorflow':
#     from .tensorflow_backend import *
#     from .tensorflow_nn import *
#     import tensorflow as tf
#     BACKEND_VERSION = tf.__version__
#     sys.stderr.write('Using TensorFlow backend.\n')

# elif BACKEND == 'mindspore':
#     from .mindspore_backend import *
#     from .mindspore_nn import *
#     import mindspore as ms
#     BACKEND_VERSION = ms.__version__
#     # set context
#     import mindspore.context as context
#     import os
#     os.environ['DEVICE_ID'] = '0'
#     context.set_context(mode=context.PYNATIVE_MODE),
#     # context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU'),
#     # enable_task_sink=True, enable_loop_sink=True)
#     # context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
#     sys.stderr.write('Using MindSpore backend.\n')

# elif BACKEND == 'paddle':
#     from .paddle_backend import *
#     from .paddle_nn import *
#     import paddle as pd
#     BACKEND_VERSION = pd.__version__
#     sys.stderr.write('Using Paddle backend.\n')
# elif BACKEND == 'torch':
#     from .torch_nn import *
#     from .torch_backend import *
#     import torch
#     BACKEND_VERSION = torch.__version__
#     sys.stderr.write('Using PyTorch backend.\n')
# else:
#     raise NotImplementedError("This backend is not supported")
