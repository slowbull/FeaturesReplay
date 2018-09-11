from __future__ import division

import os, sys, shutil, time, random, math
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import models
from torch.autograd import Variable

from sharedtensor import SharedTensor
import multiprocessing as mp

GPUS = [0, 1, 2, 3]

def main(args):
  # Init logger
  if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
  print_log('save path : {}'.format(args.save_path), log)
  state = {k: v for k, v in args._get_kwargs()}
  print_log(state, log)
  print_log("Random Seed: {}".format(args.manualSeed), log)
  print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
  print_log("torch  version : {}".format(torch.__version__), log)
  print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)

  # Init dataset
  if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

  if args.dataset == 'cifar10':
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
  elif args.dataset == 'cifar100':
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
  elif args.dataset == 'imagenet32x32':
    mean = [x / 255 for x in [122.7, 116.7, 104.0]] 
    std = [x / 255 for x in [66.4, 64.6, 68.4]]
  else:
    assert False, "Unknow dataset : {}".format(args.dataset)

  train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
     transforms.Normalize(mean, std)])
  test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

  if args.dataset == 'cifar10':
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'cifar100':
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100
  elif args.dataset == 'svhn':
    train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'stl10':
    train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
    test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
    num_classes = 10
  elif args.dataset == 'imagenet32x32':
    train_data = IMAGENET32X32(args.data_path, train=True, transform=train_transform, download=True)
    test_data = IMAGENET32X32(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 1000
  elif args.dataset == 'imagenet':
    assert False, 'Do not finish imagenet code'
  else:
    assert False, 'Do not support dataset : {}'.format(args.dataset)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, pin_memory=True)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

  print_log("=> creating model '{}'".format(args.arch), log)
  # Init model, criterion, and optimizer
  nets = models.__dict__[args.arch](args.depth, num_classes=num_classes, num_splits=args.splits)
  for i, net in enumerate(nets):
    print_log("=> network{} :\n {}".format(i,net), log)
  # define loss function (criterion) and optimizer
  criterion = torch.nn.CrossEntropyLoss()

  optimizers = []
  for i in range(args.splits):
    optimizer = torch.optim.SGD(nets[i].parameters(), state['learning_rate'], momentum=state['momentum'],
                weight_decay=state['decay'], nesterov=True)
    optimizers.append(optimizer)

  #if args.use_cuda:
  #  for net, gpu in zip(nets, args.dist_gpus):
  #    net.cuda(gpu)
  #  criterion.cuda(args.dist_gpus[-1])

  # shared tensor
  shm_lists = []
  shm_eval = SharedTensor([args.splits,])
  shm_lists.append(shm_eval)
  shm_target = SharedTensor([args.batch_size,], dtype='int64')
  shm_lists.append(shm_target)

  # there should be args.splits-1 elements in shapes
  shapes = compute_shapes(nets, args)
  #shapes = [[args.batch_size, 32, 16, 16],]   # shape for 2 splits.
  for shape in shapes:
    shm_data = SharedTensor(shape)
    shm_grad = SharedTensor(shape)
    shm_lists.append(shm_data)
    shm_lists.append(shm_grad)

  # Main loop
  print('Epoch   Train_Prec@1   Train_Prec@5    Train_Loss    Test_Prec@1   Test_Prec@5  Test_Loss   Best_Prec@1  Epoch_Time(Training)')
  processes = []
  for i in range(args.splits):
    if i == 0:
      p = mp.Process(target=train_start, args=(train_loader, test_loader, nets[i], optimizers[i], shm_lists, args, i))
    elif i < args.splits-1: 
      p = mp.Process(target=train_n, args=(train_loader, test_loader, nets[i], optimizers[i], shm_lists, args, i))
    elif i == args.splits-1:
      p = mp.Process(target=train_end, args=(train_loader, test_loader, nets[i], criterion, optimizers[i], shm_lists, args, i))
    #else:
    #  p = mp.Process(target=validate, args=(test_loader, nets, criterion, shm_lists, args))

    p.start()
    processes.append(p)

  for p in processes:
    p.join()

  log.close()

# train function for first split(forward, backward, update)
def train_start(train_loader, test_loader, model, optimizer, shm_lists, args, split_id):
  model.cuda(args.dist_gpus[split_id])
  for epoch in range(args.epochs):
    wd_time = 0.
    wg_time = 0.
    f_time = 0.
    b_time = 0.

    adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, args.learning_rate*2)
    # switch to train mode
    model.train()

    for i, (inputs, target) in enumerate(train_loader):
      while len(inputs) != args.batch_size:
        inputs_copy_len = (args.batch_size - len(inputs)) if (args.batch_size - len(inputs) <= len(inputs)) else len(inputs)
        inputs = torch.cat([inputs, inputs[0:inputs_copy_len]], 0)
        target = torch.cat([target, target[0:inputs_copy_len]], 0)

      # send target to the last processor
      shm_lists[1].send(target.cpu())

      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[0])
      #input_var = Variable(inputs, requires_grad=True)
      input_var = Variable(inputs, volatile=True)

      tm = time.time()
      # compute output
      output = model(input_var)
      f_time += time.time()-tm

      # send output.data to the next layer
      shm_lists[2*split_id+2].send(output.data.cpu())

      tm = time.time()
      #print('1 start backp, ',time.time())
      # compute gradient and do SGD step
      optimizer.zero_grad()
      model.backward()
      optimizer.step()
      b_time += time.time()-tm

      tm = time.time()
      #print('start receiving grad, ',time.time())
      # recieve output.grad from grad queue and backpropagate.
      if model.delay <= 0:
        grad = shm_lists[2*split_id+3].recv()
        if args.use_cuda:
          grad = grad.cuda(args.dist_gpus[split_id])
        model.backup(grad)
      wg_time += time.time()-tm
      #print('stop receiving grad, ',time.time())

    #print('\nsplit_id: {}, wd_time: {}, wg_time: {}, f_time: {}, b_time: {}'.format(split_id, wd_time, wg_time, f_time, b_time))
    #shm_lists[0].shm_tensor[split_id] = 1 
    #while shm_lists[0].shm_tensor[split_id] == 1:
    #  pass
    #####################################################
    # test mode
    #####################################################
    model.eval()
    for i, (inputs, target) in enumerate(test_loader):
      while len(inputs) != args.batch_size:
        inputs_copy_len = (args.batch_size - len(inputs)) if (args.batch_size - len(inputs) <= len(inputs)) else len(inputs)
        inputs = torch.cat([inputs, inputs[0:inputs_copy_len]], 0)
        target = torch.cat([target, target[0:inputs_copy_len]], 0)

      # send target to the last processor
      shm_lists[1].send(target.cpu())

      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[0])
      input_var = Variable(inputs, volatile=True)

      output = model(input_var)

      # send output.data to the next layer
      shm_lists[2*split_id+2].send(output.data.cpu())


# train function for the mid split (forward, backward, update)
def train_n(train_loader, test_loader, model, optimizer, shm_lists, args, split_id):

  model.cuda(args.dist_gpus[split_id])
  for epoch in range(args.epochs):
    wd_time = 0.
    wg_time = 0.
    f_time = 0.
    b_time = 0.

    adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, args.learning_rate)

    # switch to train mode
    model.train()

    for i in range(len(train_loader)):

      tm = time.time()
      # get inputs from previous layer, and feed it into model.
      inputs = shm_lists[2*split_id].recv()
      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[split_id])
      #inputs_var = Variable(inputs, requires_grad=True)
      inputs_var = Variable(inputs, volatile=True)
      wd_time += time.time()-tm
        
      tm = time.time()
      # compute output
      output = model(inputs_var)
      f_time += time.time()-tm

      # send output.data to the next layer
      shm_lists[2*split_id+2].send(output.data.cpu())

      tm = time.time()
      # compute gradient and do SGD step
      optimizer.zero_grad()
      model.backward()
      optimizer.step()
      b_time += time.time()-tm

      # send grad to the previous layer
      if model.delay < 0:
        shm_lists[2*split_id+1].send(model.get_grad().cpu())

      tm = time.time()
      # recieve grad from the next layer
      if model.delay <= 0:
        grad = shm_lists[2*split_id+3].recv()
        if args.use_cuda:
          grad = grad.cuda(args.dist_gpus[split_id])
        model.backup(grad)
      wg_time += time.time()-tm

    #print('\nsplit_id: {}, wd_time: {}, wg_time: {}, f_time: {}, b_time: {}'.format(split_id, wd_time, wg_time, f_time, b_time))

    #shm_lists[0].shm_tensor[split_id] = 1
    #while shm_lists[0].shm_tensor[split_id] == 1:
    #  pass
    #####################################################
    # test mode
    #####################################################
    model.eval()
    for i in range(len(test_loader)):
      # get inputs from previous layer, and feed it into model.
      inputs = shm_lists[2*split_id].recv()
      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[split_id])
      inputs_var = Variable(inputs, volatile=True)
      output = model(inputs_var)
      # send output.data to the next layer
      shm_lists[2*split_id+2].send(output.data.cpu())


# train function for the last split (forward, backward, update)
def train_end(train_loader, test_loader, model, criterion, optimizer, shm_lists, args, split_id):
  model.cuda(args.dist_gpus[split_id])

  best_top1 = -1

  for epoch in range(args.epochs):
    wd_time = 0.
    wg_time = 0.
    f_time = 0.
    b_time = 0.

    adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule, args.learning_rate)

    train_losses = AverageMeter()
    train_top1 = AverageMeter()
    train_top5 = AverageMeter()
    # switch to train mode
    model.train()

    start_time = time.time()
    for i in range(len(train_loader)):
      # receive target 
      target = shm_lists[1].recv()
      if args.use_cuda:
        target = target.cuda(args.dist_gpus[-1])
      target_var = Variable(target)

      tm = time.time()
      #print('2 start f, ',time.time())
      # compute gradient and do SGD step
      # get inputs from previous layer, and feed it into model.
      inputs = shm_lists[2*split_id].recv()
      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[-1])
      inputs_var = Variable(inputs, requires_grad=True)
      wd_time += time.time()-tm
        
      tm = time.time()
      # compute output
      output = model(inputs_var)
      loss = criterion(output, target_var)
      f_time += time.time()-tm

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      train_losses.update(loss.data[0], target.size(0))
      train_top1.update(prec1[0], target.size(0))
      train_top5.update(prec5[0], target.size(0))

      tm = time.time()
      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      b_time += time.time()-tm
      #print('2 end b, ',time.time())

      # send grad to shared memomry
      shm_lists[2*split_id+1].send(inputs_var.grad.data.cpu())
      #print('send grad, ',time.time())
    #print('\nsplit_id: {}, wd_time: {}, wg_time: {}, f_time: {}, b_time: {}'.format(split_id, wd_time, wg_time, f_time, b_time))
    #best_top1 = top1.avg if top1.avg > best_top1 else best_top1
    print('{epoch:d}       {top1.avg:.3f}       {top5.avg:.3f}       {losses.avg:.3f}       '.format(epoch=epoch, top1=train_top1, top5=train_top5, losses=train_losses), end=' ',flush=True)

    training_time = time.time() - start_time
    #shm_lists[0].shm_tensor[split_id] = 1
    #while shm_lists[0].shm_tensor[split_id] == 1:
    #  pass
    #####################################################
    # test mode
    #####################################################
    test_losses = AverageMeter()
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    model.eval()
    for i in range(len(test_loader)):
      target = shm_lists[1].recv()
      if args.use_cuda:
        target = target.cuda(args.dist_gpus[split_id])

      target_var = Variable(target, volatile=True)
      # get inputs from previous layer, and feed it into model.
      inputs = shm_lists[2*split_id].recv()
      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[-1])
      inputs_var = Variable(inputs, volatile=True)
      output = model(inputs_var)

      loss = criterion(output, target_var)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      test_losses.update(loss.data[0], inputs.size(0))
      test_top1.update(prec1[0], inputs.size(0))
      test_top5.update(prec5[0], inputs.size(0))
 
    best_top1 = test_top1.avg if test_top1.avg > best_top1 else best_top1
    print('{top1.avg:.3f}       {top5.avg:.3f}       {losses.avg:.3f}       {best_top1:.3f}       {time:.3f}'.format(top1=test_top1, top5=test_top5, losses=test_losses, best_top1=best_top1, time=training_time), flush=True)



def validate(val_loader, models, criterion, shm_lists, args):
  for model in models:
    model.cuda(args.dist_gpus[-1])
  #criterion.cuda(args.dist_gpus[-1])

  best_top1 = -1

  start_time = time.time()
  for epoch in range(args.epochs):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # wait until test time
    while shm_lists[0].shm_tensor.sum() != args.splits:
      pass
    training_time = time.time() - start_time
    # switch to evaluate mode
    for model in models:
      model.eval()

    for i, (inputs, target) in enumerate(val_loader):
      if args.use_cuda:
        inputs = inputs.cuda(args.dist_gpus[-1])
        target = target.cuda(args.dist_gpus[-1])

      input_var = Variable(inputs, volatile=True)
      target_var = Variable(target, volatile=True)

      # compute output
      output = models[0](input_var)
      for model,gpu in zip(models[1:], args.dist_gpus[1:]):
        #output = output.cuda(gpu) 
        output = model(output)

      loss = criterion(output, target_var)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
      losses.update(loss.data[0], inputs.size(0))
      top1.update(prec1[0], inputs.size(0))
      top5.update(prec5[0], inputs.size(0))

    best_top1 = top1.avg if top1.avg > best_top1 else best_top1
    print('{top1.avg:.3f}       {top5.avg:.3f}       {losses.avg:.3f}       {best_top1:.3f}       {time:.3f}'.format(top1=top1, top5=top5, losses=losses, best_top1=best_top1, time=training_time))

    shm_lists[0].shm_tensor[:] = 0
    start_time = time.time()
    ##
    #torch.cuda.empty_cache()


def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()


def compute_shapes(models, args):
  shapes = []
  # switch to evaluate mode
  for model in models:
    model.eval()

  inputs = torch.FloatTensor(1, 3, 32, 32)
  #inputs = inputs.cuda(args.dist_gpus[0])
  inputs_var = Variable(inputs, volatile=True)

  output = models[0](inputs_var)
  size = output.size()
  shape = [args.batch_size, ] + [x for x in size[1:]]
  shapes.append(shape)

  for model, gpu in zip(models[1:(args.splits-1)], args.dist_gpus[1:(args.splits-1)]):
    #output = output.cuda(gpu)
    output = model(output)
    size = output.size()
    shape = [args.batch_size, ] + [x for x in size[1:]]
    shapes.append(shape)

  return shapes


def adjust_learning_rate(optimizer, epoch, gammas, schedule, lr):
  """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
  #lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step):
      lr = lr * gamma
    else:
      break
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr

def accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

if __name__ == '__main__':
  try:
    mp.set_start_method('spawn')
  except RuntimeError:
    pass

  model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

  parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('data_path', type=str, help='Path to dataset')
  #parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
  parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'svhn', 'imagenet32x32'], help='Choose between Cifar10/100 and ImageNet.')
  parser.add_argument('--arch', metavar='ARCH', default='resnet_fr', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
  parser.add_argument('--depth', type=int, default=20, help='depth of resnet model.')
  parser.add_argument('--splits', type=int, default=2, help='splits net to multiple parts.')
  # Optimization options
  parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
  parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
  parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
  parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
  parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
  parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
  parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
  # Checkpoints
  parser.add_argument('--verbose', action='store_true', default=False, help='enable training print in the epoch')
  parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
  parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
  parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
  parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
  parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
  # Acceleration
  parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
  parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
  # random seed
  parser.add_argument('--manualSeed', type=int, help='manual seed')
  args = parser.parse_args()
  args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

  nets_per_gpu = math.ceil(args.splits / len(GPUS))
  dist_gpus = []
  cur_gpu = -1
  for i in range(args.splits):
    if i % nets_per_gpu == 0:
      cur_gpu += 1
    dist_gpus.append(GPUS[cur_gpu])
  args_var = vars(args)
  args_var['dist_gpus'] = dist_gpus
  print('GPUS distribution of nets: ', dist_gpus)

  if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
  random.seed(args.manualSeed)
  torch.manual_seed(args.manualSeed)
  if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
  cudnn.benchmark = True # find the fastest cudnn conv algorithm

  main(args)
