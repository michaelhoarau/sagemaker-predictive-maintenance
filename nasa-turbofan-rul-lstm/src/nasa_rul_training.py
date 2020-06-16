import os
import h5py as h5
import logging
import argparse
import gzip
import numpy as np
import mxnet as mx
import mxnet.gluon as G
from mxnet.gluon.contrib.estimator.event_handler import CheckpointHandler

class RULPredictionNet(G.nn.HybridSequential):
    """
    Model architecture definition
    """
    def __init__(self, hidden_size, num_layers, dropout=None, **kwargs):
        super(RULPredictionNet, self).__init__(**kwargs)
        
        self.net = G.nn.HybridSequential()
        with self.net.name_scope():
            if dropout is not None:
                self.net.add(G.rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout))
            else:
                self.net.add(G.rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers))
                
            self.net.add(G.nn.Dense(units=1))
            
    def forward(self,x):
        x = self.net(x)
        
        return x
    
class CustomCheckpointHandler(CheckpointHandler):
    def _save_params_and_trainer(self, estimator, file_prefix, *args, **kwargs):
        super(CustomCheckpointHandler, self)._save_params_and_trainer(estimator, file_prefix)
        
        # Checks if we are currently saving the best model:
        if file_prefix == self.model_prefix + '-best':
            symbol_file = os.path.join(self.model_dir, self.model_prefix + '-custom')
            estimator.net.export(symbol_file, epoch=0)
    
def preprocessing(path, batch_size):
    print('Path: ', path)
    
    train_data = os.path.join(path, 'train.h5')
    with h5.File(train_data, 'r') as ftrain:
        train_sequences = ftrain['train_sequences'][()]
        train_labels = ftrain['train_labels'][()]
        
    ftrain.close()
    
    logging.info('[### train ###] Sequences: {}'.format(train_sequences.shape))
    logging.info('[### train ###] Labels: {}'.format(train_labels.shape))
    
    X_train = mx.nd.array(train_sequences)
    y_train = mx.nd.array(train_labels)

    train_dataset = G.data.dataset.ArrayDataset(X_train, y_train)
    train_loader = G.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

def train(batch_size, epochs, learning_rate, hidden_size, num_layers, dropout, num_gpus, training_channel, model_dir):
    from mxnet.gluon.contrib.estimator import estimator as E
    from mxnet.gluon.contrib.estimator.event_handler import CheckpointHandler
    
    logging.getLogger().setLevel(logging.DEBUG)
    checkpoints_dir = '/opt/ml/checkpoints'
    checkpoints_enabled = os.path.exists(checkpoints_dir)
    
    # Preparing datasets:
    logging.info('[### train ###] Loading data')
    train_loader = preprocessing(training_channel, batch_size)
    
    # Configuring network:
    logging.info('[### train ###] Initializing network')
    net = RULPredictionNet(hidden_size, num_layers, dropout).net
    net.hybridize()
    device = mx.gpu(0) if num_gpus > 0 else mx.cpu(0)
    net.initialize(mx.init.Xavier(), ctx=device)
    
    trainer = G.Trainer(
        params=net.collect_params(),
        optimizer='adam',
        optimizer_params={'learning_rate': learning_rate},
    )
    
    # Define the estimator, by passing to it the model, 
    # loss function, metrics, trainer object and context:
    estimator = E.Estimator(
        net=net,
        loss=G.loss.L2Loss(),
        train_metrics=[mx.metric.RMSE(), mx.metric.Loss()],
        trainer=trainer,
        context=device
    )
    
    checkpoint_handler = CustomCheckpointHandler(
        model_dir=model_dir,
        model_prefix='model',
        monitor=estimator.train_metrics[0],
        mode='min',
        save_best=True
    )
    
    # Start training the model:
    logging.info('[### train ###] Training start')
    estimator.fit(train_data=train_loader, epochs=epochs, event_handlers=[checkpoint_handler])
    logging.info('[### train ###] Training end')
    
    # Cleanup model directory before SageMaker zips it to send it back to S3:
    logging.info('[### train ###] Model directory clean up, only keeps the best model')
    model_name = 'model'
    os.remove(os.path.join(model_dir, model_name + '-best.params'))
    os.remove(os.path.join(model_dir, model_name + '-symbol.json'))
    os.rename(os.path.join(model_dir, model_name + '-custom-0000.params'), os.path.join(model_dir, model_name + '-best.params'))
    os.rename(os.path.join(model_dir, model_name + '-custom-symbol.json'), os.path.join(model_dir, model_name + '-symbol.json'))
    for files in os.listdir(model_dir):
        if (files[:len(model_name + '-epoch')] == model_name + '-epoch'):
            os.remove(os.path.join(model_dir, files))
                  
    logging.info('[### train ###] Emitting metrics')
    training_rmse = estimator.train_metrics[0].get()
    training_loss = estimator.train_metrics[1].get()
    print('training rmse: {}'.format(training_rmse[1]))
    print('training loss: {}'.format(training_loss[1]))
    logging.getLogger().setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.1)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--hidden-size', type=int, default=40)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=int, default=None)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    #parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    
    logging.info('[### main ###] Start custom code')

    train(
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        num_gpus,
        args.train,
        #args.hosts, 
        #args.current_host, 
        args.model_dir
    )
    
    logging.info('[### main ###] End custom code')