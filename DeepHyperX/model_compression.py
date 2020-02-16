from DeepHyperX.pruning.iterative_pruning import *
import time
from datetime import timedelta

from DeepHyperX import models
from DeepHyperX.pruning.iterative_pruning import *
# Collection of tools to compress a neural network with.
from DeepHyperX.utils import print_memory_metrics, stop_mem_measurement, start_mem_measurement


def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of iterative pruning in the paper: '
                                                 'Learning both Weights and Connections for Efficient Neural Networks')
    parser.add_argument('--data', '-d', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names))
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-o', '--optimizer', default='SGD', metavar='O',
                        choices=optim_names,
                        help='optimizers: ' + ' | '.join(optim_names) +
                             ' (default: SGD)')
    parser.add_argument('-m', '--max_epochs', default=5, type=int,
                        metavar='E',
                        help='max number of epochs while training')
    parser.add_argument('-c', '--interval', default=5, type=int,
                        metavar='I',
                        help='checkpointing interval')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=0.005, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-t', '--topk', default=[1,5],
                        metavar='T',
                        nargs='+', type=int,
                        help='Top k precision metrics')
    parser.add_argument('--cuda', default='', action='store_true')

    return parser.parse_args()

def prune(args, model, percent, train_loader, val_loader, hyperparams, df_column_entry_dict):
    """
    Uses parameter pruning to remove connections from the model that are least relevant for the neurons.
    The concrete procedure is the traversal of the models modules, module by module (https://arxiv.org/abs/1506.02626).
    # todo elaborate on the CONCRETE pruning procedure, especially when implementing alternatives or before varying this
    Code from https://github.com/larry0123du/PyTorch-Deep-Compression
    Code explained at https://jacobgil.github.io/deeplearning/pruning-deep-learning
    :param model: the actual model (Module subclass), not a path, not just the weights
    :return: the saved model's path
    """

    # Set additional parameters required for pruning.
    # todo future work: might want to include all of these in args only, instead of passing arguments in two parameters, args and hyperparams
    hyperparams['topk'] = [1,5] # Top k precision metrics
    hyperparams['interval'] = int(args.prune_epochs) # checkpointing interval
    hyperparams['momentum'] = 0.9
    hyperparams['weight_decay'] = 0.005
    torch.cuda.empty_cache()
    print("emptied cache\n")
    print_memory_metrics("start of pruning", df_column_entry_dict)
    start_mem_measurement()
    start = time.time()
    iter_prune(args=args, train_loader=train_loader, val_loader=val_loader, the_model=model, stop_percent=percent, df_column_entry_dict=df_column_entry_dict, **hyperparams)
    time_elapse = time.time() - start

    event = 'iterative pruning'
    formatted_time = str(timedelta(seconds=time_elapse))
    df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

    print("\n"+event+" took " + formatted_time + " seconds\n")
    event = "end of pruning"
    stop_mem_measurement(event, df_column_entry_dict)
    print_memory_metrics(event, df_column_entry_dict)