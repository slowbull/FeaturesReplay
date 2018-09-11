import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import math
from collections import deque

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes * self.expansion:
          self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        
        residual = self.downsample(residual)
        out += residual
        out = self.relu(out)

        return out


class CifarResNet(object):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, block, layer_blocks, num_classes):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    #super(CifarResNet, self).__init__()
    super().__init__()

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    #print ('CifarResNet : Depth : {} '.format(depth))

    self.num_classes = num_classes

    self.layers = []

    self.prep = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
    #self.conv_1_3x3 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #self.layers.append(self.conv_1_3x3)
    #self.bn_1 = nn.BatchNorm2d(64)
    #self.layers.append(self.bn_1)
    #self.relu = nn.ReLU()
    self.layers.append(self.prep)

    list_planes = [64, 128, 256, 512]
    list_stride = [1, 2, 2, 2]
    self.inplanes = 64
    
    stage = 'stage'
    for i, planes in enumerate(list_planes):
      strides = [list_stride[i],] + [1,]*(layer_blocks[i]-1)
      cur_stage = stage + '_' + str(i+1)
      #if stride != 1 or self.inplanes != planes * block.expansion:
      #  downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
      #        nn.BatchNorm2d(planes * block.expansion),)

      # add block into self.layers
      #self.layers.append(block(self.inplanes, planes, stride, downsample))
      #setattr(self, cur_stage+'_'+str(1), self.layers[-1])
      # update self.inplanes
      for j,stride in enumerate(strides):
        # set attribute of class
        self.layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        setattr(self, cur_stage+'_'+str(j+1), self.layers[-1])

    self.avgpool = nn.AvgPool2d(4)
    self.layers.append(self.avgpool)

    self.classifier = nn.Linear(512*block.expansion, num_classes)


class CifarResNetBlock(nn.Module):
  def __init__(self, model, layers, splits_id, num_splits, delay):
    super().__init__()

    self.splits_id = splits_id
    self.num_splits = num_splits
    self.delay = delay
    self.history_inputs = deque(maxlen=delay+1)
    self.history_outputs = deque(maxlen=delay+1)
    
    self.layers = nn.Sequential(*layers)
    #splits_name = 'splits_' + str(splits_id) + '_'
    #for i, layer in enumerate(layers):
    #  setattr(self, splits_name+str(i), layer)

    if splits_id == self.num_splits - 1:
      self.classifier = model.classifier

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal(m.weight)


  def forward(self, x):
    # store inputs for backpropagation
    if self.training:
      self.history_inputs.append(x)

    #for layer in self.layers:
    #  x = layer(x)
    x = self.layers(x)

    if self.splits_id == self.num_splits-1:
      x = x.view(x.size(0), -1)
      x = self.classifier(x)

    # store outputs for backpropagation 
    #if self.training:
    #  self.history_outputs.append(x)
    return x

  def backward(self):
    if self.splits_id == self.num_splits - 1:
      #self.output.backward()
      pass
    else:
      if self.delay > 0:
        self.delay -= 1
        return

      if self.delay == 0:
        self.delay -= 1
      #prev_output = self.history_outputs.popleft()
      #prev_output.backward(self.prev_grad)
      self.forward_backward()

  def forward_backward(self):
    x = self.history_inputs[0]
    x = Variable(x.data, requires_grad=True)
    self.history_inputs[0] = x
    #for layer in self.layers:
    #  x = layer(x)
    x = self.layers(x)

    x.backward(self.prev_grad)

  def backup(self, grad):
    self.prev_grad = grad

  def get_grad(self):
    prev_input = self.history_inputs.popleft()
    return prev_input.grad.data


def resnet_bottleneck_fr(depth, num_classes=10, num_splits=1):
  if depth == 26:
    layer_blocks = [2,2,2,2] 
  elif depth == 50:
    layer_blocks = [3,4,6,3] 
  elif depth == 101:
    layer_blocks = [3,4,23,3] 
  elif depth == 152:
    layer_blocks = [3,8,36,3] 

  print ('CifarResNet : Depth : {} '.format(depth))

  model = CifarResNet(Bottleneck, layer_blocks, num_classes)
  len_layers = len(model.layers) 
  split_depth = math.ceil(len_layers / num_splits)
  nets = []
  for splits_id in range(num_splits):
    left_idx = splits_id * split_depth
    right_idx = (splits_id+1) * split_depth
    if right_idx > len_layers:
      right_idx = len_layers
    net = CifarResNetBlock(model, model.layers[left_idx:right_idx], splits_id, num_splits, num_splits - splits_id-1)
    nets.append(net) 

  return nets


