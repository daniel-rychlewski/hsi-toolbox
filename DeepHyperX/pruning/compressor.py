import os
import shutil
import time
from datetime import timedelta

import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from datetime import timedelta

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

optim_names = sorted(name for name in optim.__dict__
                     if callable(optim.__dict__[name]))
def add_timestamp():
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S.%f %Y-%m-%d ")

def adjust_learning_rate(optimizer, lr, verbose=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if verbose:
        print(optimizer.param_groups)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(model, dataloader, topk, cuda=True, df_column_entry_dict=None):
    '''
    validate the model on a given dataset
    :param
    model: specify the model to be validated
    dataloader: a loader for the dataset to be validated on
    topk: a list that specifies which top k scores we want
    cuda: whether cuda is used
    :return:
    all the top k precision scores
    '''
    scores = [AverageMeter() for _ in topk]

    # switch to evaluate mode
    model.eval()

    start = time.time()
    print(add_timestamp()+'Validating ', end='', flush=True)

    for i, (input, target) in enumerate(dataloader):
        if cuda:
            input = input.cuda()
            target = target.cuda(async=True)
        # input_var = Variable(input, volatile=True)
        # target_var = Variable(target, volatile=True)

        # compute output
        # output = model(input_var)
        output = model(input)

        # measure accuracy
        precisions = accuracy(output.data, target, topk=topk)
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        for i, s in enumerate(scores):
            s.update(precisions[i][0], input.size(0))

        if i % 20 == 0:
            print('.', end='', flush=True)

    time_elapse = time.time() - start

    event = add_timestamp() + ' compressor inference time'
    formatted_time = str(timedelta(seconds=time_elapse))
    df_column_entry_dict['Time measurement at ' + event + ' [s]'] = time_elapse

    print('\n'+event, formatted_time)
    # print(' * Prec@1 {top1.avg:.3f}% Prec@5 {top5.avg:.3f}%'
    #       .format(top1=top1, top5=top5))
    ret = list(map(lambda x:x.avg, scores))
    string = ' '.join(['Prec@%d: %.3f%%' % (topk[i], a) for i, a in enumerate(ret)])
    print(' *', string)

    # return top1.avg, top5.avg
    return ret


def save_checkpoint(state, filename='checkpoint.pth.tar', dir=None, is_best=False):
    if dir is not None and not os.path.exists(dir):
        os.makedirs(dir)
    filename = filename if dir is None else os.path.join(dir, filename)
    torch.save(state, filename)
    if is_best:
        bestname = 'model_best.pth.tar'
        if dir is not None:
            bestname = os.path.join(dir, bestname)
        shutil.copyfile(filename, bestname)


def load_checkpoint(filename='checkpoint.pth.tar', dir=None):
    assert dir is None or os.path.exists(dir)

    if dir:
        filename = os.path.join(dir, filename)

    return torch.load(filename)


def get_loaders(args):
    batch_size = args.batch_size

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    trainset = datasets.ImageFolder(
        traindir,
        transform
    )

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transform),
        batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def get_mnist_loaders(hyperparams):
    batch_size = hyperparams['batch_size']

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST(root='./data', train=True,
                              download=True, transform=transform)

    # num_workers = 0 because https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/5
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    testset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    return trainloader, testloader, testloader


def converged(old, new):
    converge = True

    for old_score, new_score in zip(old, new):
        converge = converge and abs(old_score - new_score) < 0.001

    return converge

class Compressor(object):
    def __init__(self, model, cuda=False):
        self.model = model
        self.num_layers = 0
        self.num_dropout_layers = 0
        self.dropout_rates = {}

        self.count_layers()

        self.weight_masks = [None for _ in range(self.num_layers)]
        self.bias_masks = [None for _ in range(self.num_layers)]

        self.cuda = cuda

    def count_layers(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                self.num_layers += 1
            elif isinstance(m, nn.Dropout):
                self.dropout_rates[self.num_dropout_layers] = m.p
                self.num_dropout_layers += 1

    def prune(self, args_alpha, model):
        '''
        :return: percentage pruned in the network
        '''
        index = 0
        dropout_index = 0

        num_pruned, num_weights = 0, 0

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                num = torch.numel(m.weight.data)

                alpha = float(args_alpha)
                # how to apply parameter alpha depends on model
                if model == 'he':
                    if alpha >= 0.0001:
                        conv_alpha = linear2_alpha = linear_alpha = alpha
                    else:
                        conv_alpha = 0.2
                        linear2_alpha = linear_alpha = alpha
                elif model == 'hu' or model == 'santara' or model == 'cao':
                    conv_alpha = linear2_alpha = linear_alpha = alpha
                elif model == 'luo_cnn':
                    if alpha < 3e-28:
                        linear2_alpha = conv_alpha = 0
                        linear_alpha = alpha
                    else:
                        conv_alpha = linear2_alpha = linear_alpha = alpha
                else:
                    print("warning: unknown model for fine-grained pruning, using default parameters")
                    conv_alpha = linear2_alpha = linear_alpha = alpha

                # DEBUG
                # if index == 0:
                #     print(m.weight.data.std())
                #     print("\n")

                if type(m) == nn.Conv2d or type(m) == nn.Conv1d or type(m) == nn.Conv3d:
                    alpha = conv_alpha
                else: # esp. LINEAR layers
                    if index == self.num_layers - 2:
                        alpha = linear2_alpha
                    else:
                        alpha = linear_alpha

                # use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
                # "All connections with weights below a threshold are removed from the network — converting a dense network into a sparse network, as shown in Figure 3" https://arxiv.org/pdf/1506.02626.pdf
                weight_mask = torch.ge(m.weight.data.abs(), alpha * m.weight.data.std()).type('torch.FloatTensor')
                if self.cuda:
                    weight_mask = weight_mask.cuda()
                self.weight_masks[index] = weight_mask

                # bias_mask = torch.ones(m.bias.data.size())
                bias_mask = torch.ge(m.bias.data.abs(), alpha * m.bias.data.std()).type('torch.FloatTensor')
                if self.cuda:
                    bias_mask = bias_mask.cuda()

                # for all kernels in the conv2d layer, if any kernel is all 0, set the bias to 0
                # in the case of linear layers, we search instead for zero rows
                for i in range(bias_mask.size(0)):
                    if len(torch.nonzero(weight_mask[i]).size()) == 0:
                        bias_mask[i] = 0
                self.bias_masks[index] = bias_mask

                index += 1

                layer_pruned = num - torch.nonzero(weight_mask).size(0)
                print(add_timestamp()+'number pruned in weight of layer %d: %.3f %%' % (index, 100 * (layer_pruned / num)))
                bias_num = torch.numel(bias_mask)
                bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
                print(add_timestamp()+'number pruned in bias of layer %d: %.3f %%' % (index, 100 * (bias_pruned / bias_num)))

                num_pruned += layer_pruned
                num_weights += num

                m.weight.data *= weight_mask
                m.bias.data *= bias_mask

            elif isinstance(m, nn.Dropout):
                # update the dropout rate
                mask = self.weight_masks[index - 1]
                m.p = self.dropout_rates[dropout_index] * math.sqrt(torch.nonzero(mask).size(0) \
                                             / torch.numel(mask))
                dropout_index += 1
                print(add_timestamp()+"new Dropout rate:", m.p)

        # print(self.weight_masks)
        return num_pruned / num_weights


    def set_grad(self):
        # print(self.weight_masks)
        index = 0
        for m in self.model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
                m.weight.grad.data *= self.weight_masks[index]
                m.bias.grad.data *= self.bias_masks[index]
                index += 1

