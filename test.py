import argparse
import collections
import torch
import numpy as np
import datahandler.loaders as module_loader
import datahandler.transforms as module_transform
import trainer as module_trainer
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import model.lr_scheduler as module_scheduler
import model.optimizer as module_optimizer

from parse_config import ConfigParser
from sklearn.cluster import KMeans
from tqdm import tqdm

torch.backends.cudnn.deterministic = False


def main(config):
    logger = config.get_logger('train')

    tfms = config.init_obj('transformations', [module_transform])

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', [module_loader], tfms=tfms)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', [module_arch])
    freeze_to = config['arch'].get('freeze_to', False)
    if freeze_to:
        print(freeze_to)
        model.freeze(freeze_to)

    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', [module_loss, torch.nn])
    epoch_metrics = [config.init_metric_ftn(met_dict, module_metric) for met_dict in config['metrics']['epoch']]
    running_metrics = [config.init_metric_ftn(met_dict, module_metric) for met_dict in config['metrics']['running']]

    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', [torch.optim, module_optimizer], trainable_params)

    try:
        lr_scheduler = config.init_obj('lr_scheduler', [torch.optim.lr_scheduler, module_scheduler], optimizer)
    except KeyError:
        lr_scheduler = None

    train_kwargs = {'model':model,
                    'criterion':criterion,
                    'metric_ftns':[epoch_metrics, running_metrics],
                    'optimizer':optimizer,
                    'config':config,
                    'data_loader':data_loader,
                    'valid_data_loader':valid_data_loader,
                    'lr_scheduler':lr_scheduler
                   }

    trainer = config.init_obj('trainer', [module_trainer], **train_kwargs)
    trainer.evaluate()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]

    config = ConfigParser.from_args(args, options)
    main(config)
