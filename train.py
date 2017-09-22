import argparse
from datetime import datetime
import os
import sys

import numpy as np
import torch
from torchvision import transforms
import visdom

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.datasets.egohands import EgoHandDataset
from faster_rcnn.datasets.lisa_hd import LISADataset
from faster_rcnn.datasets.imdb import concat_datasets
from faster_rcnn.scripttools import cmdutils
'''
========TO-DO========
[x] Use Visdom instead of Tensorboard
[ ] Write all training code in a train function and call that
    function from main
[ ] Incorporate resume capability, also add this for visdom
[ ] Better format the output, make them more informative
[ ] Use command line argument to parse the configuration rather
    than using CFG file
[ ] Code Clean-Up
'''

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset_name', type=str, default='lisa', help="[lisa|egohands|all]")
# Save arguments
parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default='checkpoints',
    help="Main dir where to save models")
parser.add_argument(
    '--exp_id', type=str, default='experiment', help="Name of the experiment")

# Visdom arguments
parser.add_argument(
    '--no_visdom', action="store_true", help="Deactivate visdom")
parser.add_argument(
    '--visdom_port_id',
    type=int,
    default='8990',
    help="Number of the port to use for visdom")
args = parser.parse_args()
# Print options
opts = vars(args)

print('------------ Options -------------')
for k, v in sorted(opts.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

output_dir = os.path.join(args.checkpoint_dir, args.dataset_name, args.exp_id)

# Hyper params

cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'

start_step = 0
end_step = 100000
lr_decay_steps = {60000, 80000}
lr_decay = 1. / 10

rand_seed = 1024
_DEBUG = True

# Prepare transforms
train_transform = transforms.ToTensor()

transform_params = {
    'shape': (416, 416),
    'jitter': 0.1,
    'hue': 0.1,
    'saturation': 1.5,
    'exposure': 1.5
}

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# visdom: Initialize all the plots
if not args.no_visdom:
    viz = visdom.Visdom(port=args.visdom_port_id)
    cmdutils.log_print(
        'Visdom hosted on port {:d}'.format(args.visdom_port_id))
    faster_rcnn_plot = viz.line(
        X=torch.zeros((1, )).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel='Iteration',
            ylabel='Loss',
            title='Current FasterRCNN Loss',
            legend=['RPN Loss', 'Fast RCNN Loss', 'Faster RCNN Loss']))
    if _DEBUG:
        fast_rcnn_plot = viz.line(
            X=torch.zeros((1, )).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current FastRCNN Loss',
                legend=['cls loss', 'bb_reg loss', 'fast_rcnn loss']))
        rpn_plot = viz.line(
            X=torch.zeros((1, )).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current RPN Loss',
                legend=['cls loss', 'bb_reg loss', 'total loss']))

# load data
if args.dataset_name == 'egohands' or args.dataset_name == 'all':
    egohands_dataset = EgoHandDataset(
        'train',
        transform=train_transform,
        transform_params=transform_params,
        use_cache=False)
    dataset = egohands_dataset
if args.dataset_name == 'lisa' or args.dataset_name == 'all':
    lisa_dataset = LISADataset(
        'train',
        transform=train_transform,
        transform_params=transform_params,
        use_cache=False)
    dataset = lisa_dataset
if args.dataset_name == 'all':
    dataset = concat_datasets([lisa_dataset, egohands_dataset])
else:
    raise ValueError('got dataset_name {} but expected "lisa" "egohands"\
                     or "all"'.format(args.dataset_name))

dataset.enrich_annots()
# TODO add flip annotations
# print('Loaded {} samples for training'.format(len(dataset)))

annotations = dataset.annotations

data_layer = RoIDataLayer(annotations, dataset.num_classes)

# load net
net = FasterRCNN(classes=dataset.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_pretrained_npy(net, pretrained_model)
# model_file = '/media/longc/Data/models/VGGnet_fast_rcnn_iter_70000.h5'
# model_file = 'models/saved_model3/faster_rcnn_60000.h5'
# network.load_net(model_file, net)
# exp_name = 'vgg16_02-19_13-24'
# start_step = 60001
# lr /= 10.
# network.weights_normal_init([net.bbox_fc, net.score_fc, net.fc6, net.fc7], dev=0.01)

net.cuda()
net.train()

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(
    params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if os.path.exists(output_dir):
    res = cmdutils.confirm(
        'output_dir {} already exists, keep going ?'.format(output_dir))
    if not res:
        sys.exit("Stopped execution")
else:
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()

for step in range(start_step, end_step + 1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1. / fps)
        cmdutils.log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            cmdutils.log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' %
                               (tp / fg * 100., tf / bg * 100., fg / step_cnt,
                                bg / step_cnt))
            cmdutils.log_print(
                '\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f'
                % (net.rpn.cross_entropy.data.cpu().numpy()[0],
                   net.rpn.loss_box.data.cpu().numpy()[0],
                   net.cross_entropy.data.cpu().numpy()[0],
                   net.loss_box.data.cpu().numpy()[0]))
        re_cnt = True

    # Plot on Visdom
    if not args.no_visdom and (step % log_interval == 0):
        # Plot Faster RCNN Loss
        viz.line(
            X=torch.ones((1, 3)).cpu() * step,
            Y=torch.Tensor(
                [net.rpn.loss.data[0], net.loss.data[0],
                 loss.data[0]]).unsqueeze(0),
            win=faster_rcnn_plot,
            update='append')
        if _DEBUG:
            # Plot Fast RCNN Loss
            viz.line(
                X=torch.ones((1, 3)).cpu() * step,
                Y=torch.Tensor([
                    net.cross_entropy.data[0], net.loss_box.data[0],
                    net.loss.data[0]
                ]).unsqueeze(0),
                win=fast_rcnn_plot,
                update='append')
            # Plot RPN Loss
            viz.line(
                X=torch.ones((1, 3)).cpu() * step,
                Y=torch.Tensor([
                    net.rpn.cross_entropy.data[0], net.rpn.loss_box.data[0],
                    net.rpn.loss.data[0]
                ]).unsqueeze(0),
                win=rpn_plot,
                update='append')

    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print(('save model: {}'.format(save_name)))
    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(
            params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False
