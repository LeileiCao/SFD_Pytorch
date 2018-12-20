import sys
import os
import pickle
import argparse
import scipy.io as sio
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import cv2
from torch.autograd import Variable
from data import FACEroot, BaseTransform, FACE

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import new_nms
from utils.timer import Timer
from models.SFD_net import build_net

parser = argparse.ArgumentParser(description='evaluate SFD on wider face')

parser.add_argument('-m', '--trained_model', default='weights/SFD.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/SFD_net/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = FACE

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()


def test_net(net, detector, cuda, image, transform, max_per_image=300, thresh=0.005):

    # dump predictions and assoc. ground truth to text file for now
    num_classes = 2
    all_boxes = []

    img = cv2.imread(image,cv2.IMREAD_COLOR)

    scale = torch.Tensor([img.shape[1], img.shape[0],
                         img.shape[1], img.shape[0]])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)      # forward pass
    boxes, scores = detector.forward(out,priors)
    boxes = boxes[0]
    scores=scores[0]

    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    # scale each detection back up to the image

    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > thresh)[0]
        if len(inds) == 0:
            all_boxes = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = new_nms(c_dets,0.3)
        c_dets = c_dets[keep, :]
        all_boxes = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[:, -1]])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            keep = np.where(all_boxes[:, -1] >= image_thresh)[0]
            all_boxes = all_boxes[keep, :]    
    return all_boxes



if __name__ == '__main__':
    # load net
    img_dim = 640
    num_classes = 2
    net = build_net('test')    # initialize detector
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    print(net)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()

    top_k = 400
    detector = Detect(num_classes,0,cfg)
    wider_face_mat = sio.loadmat('./eval/wider_face_val.mat')
    event_list = wider_face_mat['event_list']
    file_list = wider_face_mat['file_list']
    save_folder = args.save_folder
    image_path = '../data/widerface/WIDER_val/images/'
    rgb_means = ((104, 117, 123))
    for index, event in enumerate(event_list):
        filelist= file_list[index][0]
        im_dir= event[0][0]
        if not os.path.exists(save_folder + im_dir): os.makedirs(save_folder + im_dir)
        for num, file in enumerate(filelist):
            im_name = file[0][0]
            zipname = '%s/%s.jpg' %(im_dir,im_name)
            data = image_path+zipname
            boxlist=test_net(net, detector, args.cuda, data,
                    BaseTransform(640, rgb_means, (2, 0, 1)),
                    top_k, thresh=0.05)
            f = open(save_folder + im_dir + '/' + im_name + '.txt', 'w')
            f.write('{:s}\n'.format('%s/%s.jpg' % (im_dir,im_name)))
            f.write('{:d}\n'.format(len(boxlist)))
            for b in boxlist:
                x1,y1,x2,y2,s = b
                f.write('{:.1f} {:.1f} {:.1f} {:.1f} {:.3f}\n'.format(x1,y1,(x2-x1+1),(y2-y1+1),s))
            f.close()
            print('event:%d num:%d' % (index + 1, num + 1))
