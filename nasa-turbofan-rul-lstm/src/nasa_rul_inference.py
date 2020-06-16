#import os
import numpy as np
#import pandas as pd
#import requests
#import json
#import argparse

import mxnet as mx
import mxnet.gluon as G
import logging

# Before a model can be served, it must be loaded. The model server will load this model by 
# invoking the model_fn function from this inference script. If a model_fn function is not
# provided, the model server uses a default model_fn function. The default function works 
# with MXNet Module model objects saved via the default save function. Amazon SageMaker 
# injects the directory where your model files and sub-directories have been mounted. 
# Your model function should return a model object that can be used for model serving.
def model_fn(model_dir):
    logging.info('[### model_fn ###] Loading model from {}'.format(model_dir))
    
    #net = RULPredictionNet(hidden_size=109, num_layers=4, dropout=None).net
    #net.load_parameters('{}/model-best.params'.format(model_dir))
    
    device = mx.cpu()
    params_file = '{}/model-best.params'.format(model_dir)
    symbol_file = '{}/model-symbol.json'.format(model_dir)
    net = G.nn.SymbolBlock.imports(symbol_file=symbol_file, 
                                   input_names=['data'], 
                                   param_file=params_file, 
                                   ctx=device)

    return net

# Model serving is the process of responding to inference requests received by SageMaker 
# InvokeEndpoint API calls. Defining how to handle these requests can be done using input_fn, 
# predict_fn, and output_fn, some of which may be your own implementations. The SageMaker 
# MXNet model server breaks request handling into three steps:
# - input processing
# - prediction
# - output processing

def input_fn(request_body, request_content_type):
    logging.info('[### input_fn ###] Entering input_fn() method')
    logging.info('[### input_fn ###] request_content_type: {}'.format(request_content_type))
    logging.info('[### input_fn ###] request_body: {}'.format(type(request_body)))
    
    output = eval(request_body)
    output = np.frombuffer(output, dtype=np.float32)
    output = output.reshape((-1, 20, 17))
    
    output = mx.nd.array(output)
    #output = output.expand_dims(axis=0)
    #output = output.asnumpy()
    
    #return mx.io.NDArrayIter(output, label=None, batch_size=1)
    logging.info('[### input_fn ###] new output_shape: {}'.format(output.shape))
    return output

def predict_fn(input_object, model):
    logging.info('[### predict_fn ###] Entering predict_fn() method')
    return model.forward(input_object)

def output_fn(prediction, content_type):
    logging.info('[### output_fn ###] Entering output_fn() method')
    logging.info('[### output_fn ###] prediction: {}'.format(type(prediction)))
    logging.info('[### output_fn ###] prediction.shape: {}'.format(prediction.shape))
    preds = []
    for p in prediction.asnumpy():
        logging.info('[### output_fn ###] p: {}'.format(p[0]))
        preds.append(p[0])
        
    return str(preds)