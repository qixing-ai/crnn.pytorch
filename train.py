from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
# from warpctc_pytorch import CTCLoss
from torch.nn import CTCLoss
import os
from tool import utils, dataset

import models.crnn as net
import params

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--trainroot', required=False,default='./trainlmdb', help='path to train dataset')
parser.add_argument('-val', '--valroot', required=False, default='./trainlmdb',help='path to val dataset')
args = parser.parse_args()

# 创建模型保存文件件
if not os.path.exists(params.expr_dir):
    os.makedirs(params.expr_dir)

# 确保每次随机都是一样的
random.seed(params.manualSeed)
np.random.seed(params.manualSeed)
torch.manual_seed(params.manualSeed)
# 增加程序的运行效率
cudnn.benchmark = True

if torch.cuda.is_available() and not params.cuda:
    print("警告:你有一个CUDA设备，所以你应该将CUDA的params.py设置为True")

# -----------------------------------------------
"""
加载训练和验证数据
"""
def data_loader():
    # train
    train_dataset = dataset.lmdbDataset(root=args.trainroot)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.randomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batchSize, \
                                               shuffle=True, sampler=sampler, num_workers=int(params.workers), \
                                               collate_fn=dataset.alignCollate(imgH=params.imgH, imgW=params.imgW, keep_ratio=params.keep_ratio))
    
    # val
    val_dataset = dataset.lmdbDataset(root=args.valroot, transform=dataset.resizeNormalize((params.imgW, params.imgH)))
    assert val_dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
    return train_loader, val_loader

train_loader, val_loader = data_loader()

# -----------------------------------------------
"""
网络初始化
重初始化
加载预训练模型
"""
def weights_init(m):
    classname = m.__class__.__name__
    # 对卷积网络，m是模型，weight是权重
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # 对回归网络
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def net_init():
    nclass = len(params.alphabet) + 1
    crnn = net.CRNN(params.imgH, params.nc, nclass, params.nh)
    # CRNN网络中每个子模块都执行weights_init
    crnn.apply(weights_init)

    # 加载预训练
    if params.pretrained != '':
        print('加载预训练模型 %s' % params.pretrained)
        if params.multi_gpu:
            crnn = torch.nn.DataParallel(crnn)
        crnn.load_state_dict(torch.load(params.pretrained))
    
    return crnn

crnn = net_init()
# 打印
print(crnn)

# -----------------------------------------------
"""
初始化在utils.py中定义的一些utils
"""
# 计算平均 for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.averager()

# 在str和label之间进行转换.
converter = utils.strLabelConverter(params.alphabet)

# -----------------------------------------------
"""
计算丢失率
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
初始化一些张量
把张量和网放在cuda上
    NOTE:
        图像，文本，长度都被val和train使用
        becaues train and val 永远不会同时使用它。
"""
image = torch.FloatTensor(params.batchSize, 3, params.imgH, params.imgH)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()
    crnn = crnn.cuda()
    if params.multi_gpu:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(params.ngpu))
# data：存储了Tensor，是本体的数据
# grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
# grad_fn：指向Function对象，用于反向传播的梯度计算之用
image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
设置优化器
"""
if params.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    处理lossnan
    NOTE:
        我用不同的方式来处理亏损南根据torch的版本. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity = True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        crnn.register_backward_hook(crnn.backward_hook)

# -----------------------------------------------

def val(net, criterion):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    val_iter = iter(val_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager() # The blobal loss_avg is used by train

    max_iter = len(val_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for i in cpu_texts:
            cpu_texts_decode.append(i.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * params.batchSize)
    print('Val loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

# 开始训练
def train(net, criterion, optimizer, train_iter):
    for p in crnn.parameters():
        p.requires_grad = True

    # model.train() ：启用 BatchNormalization 和 Dropout
    # model.eval() ：不启用 BatchNormalization 和 Dropout
    crnn.train()
    data = train_iter.next()
    cpu_images, cpu_texts = data
    # 图片数量
    batch_size = cpu_images.size(0)
    # 将图片加载到image
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)
    
    optimizer.zero_grad()
    # 跑crnn
    preds = crnn(image)
    # tensor变成variable之后才能进行反向传播求梯度?用变量.backward()进行反向传播之后,var.grad中保存了var的梯度
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    # crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost


if __name__ == "__main__":
    # 循环nepoch次数
    for epoch in range(params.nepoch):
        # 获取数据
        train_iter = iter(train_loader)
        i = 0
        print(len(train_loader))
        # 开启一个小循环
        while i < len(train_loader):
            cost = train(crnn, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1
            # 打印
            if i % params.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                      (epoch, params.nepoch, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()
            # 验证
            if i % params.valInterval == 0:
                val(crnn, criterion)

            # 记录检查点
            if i % params.saveInterval == 0:
                torch.save(crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(params.expr_dir, epoch, i))
