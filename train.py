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
#-----------------

import os
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

# >>> remove termcolor dependency,
#     better format the output
try:
    from termcolor import cprint
except ImportError:
    cprint = None

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)
# <<<

# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
output_dir = 'models/saved_model3'

start_step = 0
end_step = 100000
lr_decay_steps = {60000, 80000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True

# >>> redundant tensorboard dependecies; remove them.
# use_tensorboard = False
# remove_all_log = False   # remove all historical experiments in TensorBoard
# exp_name = None # the previous experiment name in TensorBoard
# <<<

use_visdom = True
port_id = 8990      # port-id for visdom

try:
    import visdom
except ImportError:
    use_visdom = False

# ------------

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
if use_visdom:
    viz = visdom.Visdom(port=port_id)
    log_print('Visdom hosted on port {:d}'.format(port_id))
    faster_rcnn_plot = viz.line(
            X = torch.zeros((1,)).cpu(),
            Y = torch.zeros((1,3)).cpu(),
            opts = dict(
                xlabel = 'Iteration',
                ylabel = 'Loss',
                title = 'Current FasterRCNN Loss',
                legend = ['RPN Loss', 'Fast RCNN Loss', 'Faster RCNN Loss']
                )
        )
    if _DEBUG:
        fast_rcnn_plot = viz.line(
                    X = torch.zeros((1,)).cpu(),
                    Y = torch.zeros((1,3)).cpu(),
                    opts = dict(
                            xlabel = 'Iteration',
                            ylabel = 'Loss',
                            title = 'Current FastRCNN Loss',
                            legend = ['cls loss', 'bb_reg loss', 'fast_rcnn loss']
                            )
                )
        rpn_plot = viz.line(
                    X = torch.zeros((1,)).cpu(),
                    Y = torch.zeros((1,3)).cpu(),
                    opts = dict(
                            xlabel = 'Iteration',
                            ylabel = 'Loss',
                            title = 'Current RPN Loss',
                            legend = ['cls loss', 'bb_reg loss', 'total loss']
                            )
                )


# load data
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
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
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):

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
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True

    # Plot on Visdom
    if use_visdom and (step % log_interval == 0):
        # Plot Faster RCNN Loss
        viz.line(
            X = torch.ones((1,3)).cpu() * step,
            Y = torch.Tensor([net.rpn.loss.data[0], net.loss.data[0], loss.data[0]]).unsqueeze(0),
            win = faster_rcnn_plot,
            update = 'append'
        )
        if _DEBUG:
            # Plot Fast RCNN Loss
            viz.line(
                X = torch.ones((1,3)).cpu() * step,
                Y = torch.Tensor([net.cross_entropy.data[0], net.loss_box.data[0],
                             net.loss.data[0]]).unsqueeze(0),
                win = fast_rcnn_plot,
                update = 'append'
            )
            # Plot RPN Loss
            viz.line(
                X = torch.ones((1,3)).cpu() * step,
                Y = torch.Tensor([net.rpn.cross_entropy.data[0], net.rpn.loss_box.data[0],
                                 net.rpn.loss.data[0]]).unsqueeze(0),
                win = rpn_plot,
                update = 'append'
            )

    if (step % 10000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print(('save model: {}'.format(save_name)))
    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

